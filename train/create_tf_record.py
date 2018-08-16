# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the CORL2017ImitationLearningData to TFRecord.

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import io
import logging
import os
import random
import glob
import h5py

import contextlib2
import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')

FLAGS = flags.FLAGS

STEER = 0
GAS = 1
BRAKE = 2
SPEED = 10
COMMAND = 24
GAME_TIME = 20
DELTA_TIME = 67 # milisecond

def is_next_time(row_data, next_row):
  return abs(row_data[GAME_TIME]+DELTA_TIME-next_row[GAME_TIME]) < 2


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.
  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards
  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords

# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir

  logging.info('Reading from dataset.')

  files = sorted(glob.glob(os.path.join(data_dir, 'data*.h5')))

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)

  idx = -1
  output_filename = os.path.join(FLAGS.output_dir,
                             'data.record')

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = open_sharded_output_tfrecords(
         tf_record_close_stack, output_filename, FLAGS.num_shards)
    for index in range(len(files)):
      print(files[index])
      data = h5py.File(files[index], 'r')
      for row in range(data['targets'].shape[0]):
        idx += 1
        img = data['rgb'][row]
        row_data = data['targets'][row]
        speed = row_data[SPEED]
        command = int(row_data[COMMAND])
        if command in [0, 2]:
          command = 0 # follow
        elif command == 5:
          command = 1 # straight
        elif command == 3:
          command = 2 # left
        elif command == 4:
          commandn = 3 # right
        else:
          assert(False)
        label = row_data[np.array([STEER, GAS, BRAKE])]
        speed_out = speed
        if row < data['targets'].shape[0]-1 and is_next_time(row_data, data['targets'][row+1]):
          speed_out = data['targets'][row+1][SPEED]
        elif row == row < data['targets'].shape[0]-1 and index < len(files)-1:
          next_data = h5py.File(files[index+1], 'r')
          next_row = next_data['targets'][0]
          if is_next_time(row_data, next_row):
            speed_out = next_row[SPEED]

        try:
          feature_dict = {
            'image': int64_list_feature(np.reshape(img, [-1]).tolist()),
            'command': int64_feature(command),
            'speed': float_feature(speed),
            'label': float_list_feature(label.tolist()),
            'speed_out': float_feature(speed_out)
          }
          tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

          shard_idx = idx % FLAGS.num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        except ValueError:
          logging.warning('Invalid example: %s, ignoring.', xml_path)

  print('total is %s' % idx)

if __name__ == '__main__':
  tf.app.run()
