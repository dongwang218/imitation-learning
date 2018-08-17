import os
import random
import glob
import h5py
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

HEIGHT = 88
WIDTH = 200
SPEED_DIV = 25.0
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def parse_fn(string_record, is_training=False):
  example = tf.parse_single_example(string_record,
                                    features={
                                      'command': tf.FixedLenFeature([1], tf.int64),
                                      'speed': tf.FixedLenFeature([1], tf.float32),
                                      'speed_out': tf.FixedLenFeature([1], tf.float32),
                                      'image': tf.VarLenFeature(tf.int64),
                                      'label': tf.FixedLenFeature([3], tf.float32)
                                    })
  image = tf.sparse_tensor_to_dense(example['image'])
  image = tf.multiply(tf.cast(tf.reshape(image, shape=[HEIGHT, WIDTH, 3]), tf.float32), 1.0/255)
  if is_training:
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.2, dtype=tf.float32)
    image = tf.add(image, noise)
    image = tf.clip_by_value(image, 0, 1)
  steer, gas, brake = tf.split(example['label'], 3, axis=-1)
  speed = tf.multiply(example['speed'], 1/SPEED_DIV)
  speed_out = tf.multiply(example['speed_out'], 1/SPEED_DIV)

  return {'image': image, 'command': example['command'], 'speed': speed}, {'branch_speed': speed_out, 'steer': steer, 'gas': gas, 'brake': brake}

num_parallel_readers=10
shuffle_buffer_size=1000
prefetch_buffer_size=1000
def input_fn(data_dir, batch_size, is_training):
  files = tf.data.Dataset.list_files(os.path.join(data_dir, 'data.record-*'))
  dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=num_parallel_readers))
  dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
  dataset = dataset.apply(tf.contrib.data.map_and_batch(
    map_func=lambda value: parse_fn(value, is_training), batch_size=batch_size))
  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
  return dataset

def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.
  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  x = keras.layers.add([x, input_tensor])
  x = Activation('relu')(x)
  return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,
                                                                          2)):
  """A block that has a conv layer at shortcut.
  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the first conv layer in the block.
  Returns:
      Output tensor for the block.
  Note that from stage 3,
  the first conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(
      filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(
          input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  shortcut = Conv2D(
      filters3, (1, 1), strides=strides, name=conv_name_base + '1')(
          input_tensor)
  shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

  x = keras.layers.add([x, shortcut])
  x = Activation('relu')(x)
  return x

def simple_resnet(img_input, weights):
  bn_axis = 3

  x = keras.layers.Conv2D(
      64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
  x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

  x = keras.layers.GlobalAveragePooling2D()(x)

  model = keras.Model(img_input, x, name='resnet50')

  # load weights
  if weights == 'imagenet':
    weights_path = keras.utils.get_file(
        'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        md5_hash='a268eb855778b3df3c7506639542a6af')
    model.load_weights(weights_path)
  return model

def create_model(args):
  inputs = keras.Input(shape=(32,))  # Returns a placeholder tensor

  img_input = keras.Input(shape=(HEIGHT, WIDTH, 3), dtype='float32', name='image')
  speed_input = keras.Input(shape=(1,), dtype='float32', name='speed')
  command_input = keras.Input(shape=(1,), dtype='int32', name='command')

  resnet = simple_resnet(img_input, args.weights)
  img_output = keras.layers.Flatten()(resnet.output)

  with tf.name_scope('speed'):
    speed = keras.layers.Dense(64, activation='relu')(speed_input)
    speed = keras.layers.Dropout(0.3)(speed)
    speed = keras.layers.Dense(128, activation='relu')(speed)
    speed = keras.layers.Dropout(0.3)(speed)

  j = keras.layers.Concatenate()([img_output, speed])
  j = keras.layers.Dense(512, activation='relu')(j)
  j = keras.layers.BatchNormalization()(j)
  together = keras.layers.Dropout(0.3)(j)

  branches = ['follow', 'staight', 'left', 'right']
  outputs = []
  masked_outputs = []
  for branch_index, b in enumerate(branches):
    with tf.name_scope('branch_%s' % b):
      i = keras.layers.Dense(256, activation='relu')(together)
      i = keras.layers.BatchNormalization()(i)
      i = keras.layers.Dropout(0.3)(i)
      o = keras.layers.Dense(3, name=b)(i)
      outputs.append(o)
      this_mask = keras.layers.Lambda(lambda x: keras.backend.cast(keras.backend.equal(x, branch_index), tf.float32), output_shape=[1])(command_input)
      masked = keras.layers.multiply([o, this_mask])
      masked_outputs.append(masked)
  with tf.name_scope('branch_%s' % 'speed'):
    i = keras.layers.Dense(256, activation='relu')(together)
    i = keras.layers.BatchNormalization()(i)
    i = keras.layers.Dropout(0.3)(i)
    outputs.append(keras.layers.Dense(1, name='branch_speed')(i))

  masked = keras.layers.add(masked_outputs)
  outputs.append(keras.layers.Lambda(lambda x: x[:, 0:1], output_shape=[1], name='steer')(masked))
  outputs.append(keras.layers.Lambda(lambda x: x[:, 1:2], output_shape=[1], name='gas')(masked))
  outputs.append(keras.layers.Lambda(lambda x: x[:, 2:3], output_shape=[1], name='brake')(masked))

  return keras.Model(inputs = [img_input, command_input, speed_input], outputs = outputs)

def create_callbacks(model, args):
  callbacks = []

  # save the prediction model
  if args.snapshots:
    # ensure directory created first; otherwise h5py will error after epoch.
    if not os.path.exists(args.snapshot_path):
      os.makedirs(args.snapshot_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
      os.path.join(
        args.snapshot_path,
        'imitation_{dataset_type}_{{epoch:02d}}.h5'.format(dataset_type='resnet')
      ),
      verbose=1
    )
    callbacks.append(checkpoint)

  lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
  callbacks.append(lr_scheduler)

  logging = keras.callbacks.TensorBoard(log_dir=args.log_dir)
  callbacks.append(logging)

  return callbacks

def parse_args():
  parser     = argparse.ArgumentParser(description='Simple training script.')
  parser.add_argument('--snapshot', help='Snapshot to resume training with.')
  parser.add_argument('--weights',  help='Weights to use for initialization (defaults to \'imagenet\').', default='imagenet')

  parser.add_argument('--batch-size',    help='Size of the batches.', default=120, type=int)
  parser.add_argument('--epochs',        help='Number of epochs to train.', type=int, default=50)
  parser.add_argument('--snapshot-path', help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
  parser.add_argument('--log-dir', help='Path to store logs of models during training', default='./logs')
  parser.add_argument('--no-snapshots',  help='Disable saving snapshots.', dest='snapshots', action='store_false')
  parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
  parser.add_argument('--train-dir', help='wher images are.', default='data/train')
  parser.add_argument('--val-dir', help='wher images are.', default='data/val')
  parser.add_argument('--lr', help='learning rate', default = 0.0002, type=float)

  args = parser.parse_args()
  if args.snapshot is not None:
    args.weights = None

  return args

def train(args):

  # tf.enable_eager_execution()

  train_data = input_fn(args.train_dir, args.batch_size, True)
  valid_data = input_fn(args.val_dir, args.batch_size, False)
  # training_iterator = train_data.make_one_shot_iterator()
  # validation_iterator = valid_data.make_initializable_iterator()

  size_training = 657599
  steps = size_training // args.batch_size

  if args.snapshot:
    print('Loading model, this may take a second...')
    model            = keras.models.load_model(args.snapshot, custom_objects=custom_objects)
  else:
    print('Creating model, this may take a second...')
    model = create_model(args)

  losses = {
    'steer': 'mean_squared_error',
    'gas': 'mean_squared_error',
    'brake': 'mean_squared_error',
    "branch_speed": 'mean_squared_error'
  }
  lossWeights = {
    'steer': 0.95*0.45,
    'gas': 0.95*0.45,
    'brake': 0.95*0.05,
    "branch_speed": 0.05
  }
  print(model.summary())


  # The compile step specifies the training configuration.
  adam = keras.optimizers.Adam(args.lr)
  optimizer = adam
  model.compile(optimizer=optimizer,
                loss=losses, loss_weights=lossWeights,
                metrics=['mse'])

  callbacks = create_callbacks(model, args)

  # Trains for 5 epochs
  model.fit(train_data, epochs=args.epochs, callbacks=callbacks, validation_data=valid_data, steps_per_epoch=steps)

if __name__ == '__main__':
  args = parse_args()
  train(args)
