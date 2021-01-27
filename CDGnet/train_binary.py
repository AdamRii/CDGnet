

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os

from absl import app
from absl import flags

import train

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_directory',
    '/data/wenzh/fangfeiteng/vicsek/N300_L5_ETA1.5',
    'Directory which contains the train and test datasets.')
flags.DEFINE_integer(
    'time_index',
    99,
    'The time index of the target mobilities.')
flags.DEFINE_integer(
    'max_files_to_load',
    50,
    'The maximum number of files to load from the train and test datasets.')
flags.DEFINE_string(
    'checkpoint_path',
    'va1.5/t99f50.ckpt',
    'Path used to store a checkpoint of the best model.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_file_pattern = os.path.join(FLAGS.data_directory, 'train/Con*')
  test_file_pattern = os.path.join(FLAGS.data_directory, 'test/Con*')
  train.train_model(
      train_file_pattern=train_file_pattern,
      test_file_pattern=test_file_pattern,
      max_files_to_load=FLAGS.max_files_to_load,
      time_index=FLAGS.time_index,
      checkpoint_path=FLAGS.checkpoint_path)


if __name__ == '__main__':
  app.run(main)
