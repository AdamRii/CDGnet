

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import pickle

from absl import logging
import enum
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from typing import Any, Dict, List, Optional, Text, Tuple, Sequence
from tensorflow.python import debug as tf_debug
import graph_model

tf.enable_resource_variables()

LossCollection = collections.namedtuple('LossCollection',
                                        'l1_loss, l2_loss')
GlassSimulationData = collections.namedtuple('GlassSimulationData',
                                             'positions, targets, types, box')


def get_targets(
        initial_positions,
        trajectory_target_positions):

    targets = np.mean([np.linalg.norm(t - initial_positions, axis=-1)
                       for t in trajectory_target_positions], axis=0)

    return targets.astype(np.float32)


def load_data(
        file_pattern,
        time_index,
        max_files_to_load=None):
    """Returns a dictionary containing the training or test dataset.
    Args:
      file_pattern: pattern matching the files with the simulation data.
      time_index: the time index of the targets.
      max_files_to_load: the maximum number of files to load.
    """
    filenames = tf.io.gfile.glob(file_pattern)
    if max_files_to_load:
        filenames = filenames[:max_files_to_load]

    static_structures = []
    for filename in filenames:
        with tf.io.gfile.GFile(filename, 'rb') as f:
            data = pickle.load(f)
        box = np.array(data['box'])
        targets=[data['va'][time_index].astype(np.float32)]
        static_structures.append(GlassSimulationData(
            positions=data['positions'].astype(np.float32),

            targets=np.array(targets),
            types=data['angles'].astype(np.float32),

            box=box.astype(np.float32)))

    return static_structures


def get_loss_ops(
        prediction,
        target):
    return LossCollection(l1_loss=tf.reduce_mean(tf.abs(prediction - target)),l2_loss=tf.reduce_mean((prediction - target)**2))

def get_minimize_op(
        loss,
        learning_rate,
        grad_clip=None):

    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    
    if grad_clip:
        grads, _ = tf.clip_by_global_norm([g for g, _ in grads_and_vars], grad_clip)
        grads_and_vars = [(g, pair[1]) for g, pair in zip(grads, grads_and_vars)]
    
    minimize = optimizer.apply_gradients(grads_and_vars)
    return minimize


def _log_stats_and_return_mean_correlation(
        label,
        stats):

    for key in LossCollection._fields:
        values = [getattr(s, key) for s in stats]
        mean = np.mean(values)
        std = np.std(values)
        logging.info('%s: %s: %.4f +/- %.4f', label, key, mean, std)
    return np.mean([s.l1_loss for s in stats])


def train_model(train_file_pattern,
                test_file_pattern,
                max_files_to_load=None,
                n_epochs=1000,
                time_index=9,
                augment_data_using_rotations=True,
                learning_rate=1e-4,
                grad_clip=1.0,
                n_recurrences=7,
                mlp_sizes=(30,15,1),
                mlp_kwargs=None,
                edge_threshold=1,
                measurement_store_interval=1000,
                checkpoint_path=None):

    if mlp_kwargs is None:
        mlp_kwargs = dict(initializers=dict(w=tf.variance_scaling_initializer(1.0),
                                            b=tf.variance_scaling_initializer(0.1)))
    # Loads train and test dataset.
    dataset_kwargs = dict(
        time_index=time_index,
        max_files_to_load=max_files_to_load)
    training_data = load_data(train_file_pattern, **dataset_kwargs)
    test_data = load_data(test_file_pattern, **dataset_kwargs)

    # Defines wrapper functions, which can directly be passed to the
    # tf.data.Dataset.map function.
    def _make_graph_from_static_structure(static_structure):
        """Converts static structure to graph, targets and types."""
        return (graph_model.make_graph_from_static_structure(
            static_structure.positions,
            static_structure.types,
            static_structure.box,
            edge_threshold),
                static_structure.targets,
                static_structure.types)

    def _apply_random_rotation(graph, targets, types):
        """Applies random rotations to the graph and forwards targets and types."""
        return graph_model.apply_random_rotation(graph), targets, types


    placeholders = GlassSimulationData._make(
        tf.placeholder(s.dtype, (None,) + s.shape) for s in training_data[0])
    dataset = tf.data.Dataset.from_tensor_slices(placeholders)
    dataset = dataset.map(_make_graph_from_static_structure)
    dataset = dataset.cache()
    dataset = dataset.shuffle(400)
    # Augments data. This has to be done after calling dataset.cache!
    if augment_data_using_rotations:
        dataset = dataset.map(_apply_random_rotation)
    dataset = dataset.repeat()
    train_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.Dataset.from_tensor_slices(placeholders)
    dataset = dataset.map(_make_graph_from_static_structure)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    test_iterator = dataset.make_initializable_iterator()


    dataset_handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        dataset_handle, train_iterator.output_types, train_iterator.output_shapes)
    graph, targets, types = iterator.get_next()
    

    model = graph_model.GraphBasedModel(
        n_recurrences, mlp_sizes, mlp_kwargs)
    prediction = model(graph)
    print('prediction')
    print(prediction)
    print('targets')
    print(targets)
    # Defines loss and minimization operations.
    loss_ops = get_loss_ops(prediction, targets, types)
    print('loss')
    print(loss_ops)
    minimize_op = get_minimize_op(loss_ops.l1_loss, learning_rate, grad_clip)

    best_so_far = 1
    train_stats = []
    test_stats = []

    saver = tf.train.Saver()

    with tf.train.SingularMonitoredSession() as session:
        train_handle = session.run(train_iterator.string_handle())
        test_handle = session.run(test_iterator.string_handle())
        feed_dict = {p: [x[i] for x in training_data]
                     for i, p in enumerate(placeholders)}
        session.run(train_iterator.initializer, feed_dict=feed_dict)
        feed_dict = {p: [x[i] for x in test_data]
                     for i, p in enumerate(placeholders)}
        session.run(test_iterator.initializer, feed_dict=feed_dict)

        # Trains model using stochatic gradient descent on the training dataset.
        n_training_steps = len(training_data) * n_epochs
        for i in range(n_training_steps):
            feed_dict = {dataset_handle: train_handle}
            train_loss, _ = session.run((loss_ops, minimize_op), feed_dict=feed_dict)
            train_stats.append(train_loss)

            if (i + 1) % measurement_store_interval == 0:
                # Evaluates model on test dataset.
                for _ in range(len(test_data)):
                    feed_dict = {dataset_handle: test_handle}
                    test_stats.append(session.run(loss_ops, feed_dict=feed_dict))

                # Outputs performance statistics on training and test dataset.
                _log_stats_and_return_mean_correlation('Train', train_stats)
                correlation = _log_stats_and_return_mean_correlation('Test', test_stats)
                train_stats = []
                test_stats = []

                # Updates best model based on the observed correlation on the test
                # dataset.
                if correlation < best_so_far:
                    best_so_far = correlation
                    if checkpoint_path:
                        saver.save(session.raw_session(), checkpoint_path)

    return best_so_far


def apply_model(checkpoint_path,
                file_pattern,
                max_files_to_load=None,
                time_index=9):
    """Applies trained GraphModel using tensorflow.
  
    Args:
      checkpoint_path: path from which the model is loaded.
      file_pattern: pattern matching the files with the data.
      max_files_to_load: the maximum number of files to load.
        If None, all files will be loaded.
      time_index: the time index (0-9) of the target mobilities.
  
    Returns:
      Predictions of the model for all files.
    """
    dataset_kwargs = dict(
        time_index=time_index,
        max_files_to_load=max_files_to_load)
    data = load_data(file_pattern, **dataset_kwargs)

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
    graph = tf.get_default_graph()

    placeholders = GlassSimulationData(
        positions=graph.get_tensor_by_name('Placeholder:0'),
        targets=graph.get_tensor_by_name('Placeholder_1:0'),
        types=graph.get_tensor_by_name('Placeholder_2:0'),
        box=graph.get_tensor_by_name('Placeholder_3:0'))
    prediction_tensor = graph.get_tensor_by_name('Graph_1/Squeeze:0')

    dataset_handle = graph.get_tensor_by_name('Placeholder_4:0')
    test_initalizer = graph.get_operation_by_name('MakeIterator_1')
    test_string_handle = graph.get_tensor_by_name('IteratorToStringHandle_1:0')

    with tf.Session() as session:
        saver.restore(session, checkpoint_path)
        handle = session.run(test_string_handle)
        feed_dict = {p: [x[i] for x in data] for i, p in enumerate(placeholders)}
        session.run(test_initalizer, feed_dict=feed_dict)
        predictions = []
        correlations = []
        for _ in range(len(data)):
            p = session.run((prediction_tensor),
                               feed_dict={dataset_handle: handle})
            predictions.append(p)
    return predictions
