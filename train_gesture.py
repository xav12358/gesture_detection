#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of DNNClassifier for Iris plant dataset.

This example uses APIs in Tensorflow 1.4 or above.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np


FEATURE_KEYS = [] # vx and vy data
# FEATURE_KEYS_SIZE = 16

GESTURE_TRAINING = './data/record_training_random.csv'
GESTURE_TEST = './data/record_test.csv'
def input_fn(file_name, num_data, batch_size, is_training):
  """Creates an input_fn required by Estimator train/evaluate."""
  # If the data sets aren't stored locally, download them.

  def _parse_csv(rows_string_tensor):
    """Takes the string input tensor and returns tuple of (features, labels)."""
    # Last dim is the label.
    num_features = len(FEATURE_KEYS)
    num_columns = num_features + 1

    columns = tf.decode_csv(rows_string_tensor,
                            record_defaults=[[]] * num_columns)
    features = dict(zip(FEATURE_KEYS, columns[:num_features]))
    labels = tf.cast(columns[num_features], tf.int32)
    return features, labels

  def _input_fn():

    """The input_fn."""
    dataset = tf.data.TextLineDataset([file_name])
    # Skip the first line (which does not have data).
    dataset = dataset.skip(1)
    dataset = dataset.map(_parse_csv)
    if is_training:
      # For this small dataset, which can fit into memory, to achieve true
      # randomness, the shuffle buffer size is set as the total number of
      # elements in the dataset.
      dataset = dataset.shuffle(num_data)
      dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return _input_fn


feature_spec = {'x': tf.FixedLenFeature([32],tf.float32)}

def serving_input_receiver_fn():
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input_tensors')
  receiver_tensors = {'inputs': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def main(unused_argv):



  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=GESTURE_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=GESTURE_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[32])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=4,
                                          model_dir="./model/")

 


# Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.

  classifier.train(input_fn=train_input_fn, steps=400)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


  new_samples = np.array(
      test_set.data, dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)


  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  indice = 0
  for i in test_set.target:
      print('Expected {0}   predicted {1} '.format(i,predicted_classes[indice]))
      indice = indice+1


  classifier.export_savedmodel('./model', serving_input_receiver_fn)
  # saver = tf.train.Saver()
  # with tf.Session() as sess:
  #   # Restore variables from disk.
  #   saver.save(sess, "./model3/model.ckpt")
    
if __name__ == '__main__':
  tf.app.run()
