

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v1 as tf

import train


class TensorflowTrainTest(tf.test.TestCase):

  def test_apply_model(self):
    """Tests if we can apply a model to a small test dataset."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'va0.1',
                                   't0f50.ckpt')
    file_pattern = os.path.join(os.path.dirname(__file__), 'testdata',
                                'Con100_η0.1N300L5_100')
    predictions = train.apply_model(checkpoint_path=checkpoint_path,
                                    file_pattern=file_pattern,
                                    time_index=0)
    data = train.load_data(file_pattern, 0)
    targets = data[0].targets
    # correlation_value = np.corrcoef(predictions[0], targets)[0, 1]
    print('predictions:',predictions)
    print('targets:',targets) 
    # self.assertGreater(correlation_value, 0.5)


if __name__ == '__main__':
  tf.test.main()
