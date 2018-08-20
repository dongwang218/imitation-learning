import argparse
import logging
import os, sys
import glob
import numpy as np
import h5py

sys.path.append('/host/workspace/robotics/carla/PythonClient')
from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import CoRL2017

from agents.imitation.imitation_learning import ImitationLearning

SPEED = 10
COMMAND = 24
STEER = 0
GAS = 1
BRAKE = 2
def get_mae(m, n):
  return (m-n)**2

if (__name__ == '__main__'):
  argparser = argparse.ArgumentParser(description=__doc__)
  argparser.add_argument(
      '-v', '--verbose',
      action='store_true',
      dest='debug',
      help='print debug information')

  argparser.add_argument(
      '--avoid-stopping',
      default=True,
      action='store_false',
      help=' Uses the speed prediction branch to avoid unwanted agent stops'
  )
  argparser.add_argument(
      '--new-model',
      default=False,
      action='store_true',
      help=' Uses the new model'
  )
  argparser.add_argument(
      '--data-dir', type=str, required=True,
      help=' Uses the new model'
  )

  args = argparser.parse_args()

  if args.new_model:
    memory_fraction = 0.75
  else:
    memory_fraction = 0.25
  agent = ImitationLearning('city', args.avoid_stopping, new_model=args.new_model, memory_fraction=memory_fraction)

  files = sorted(glob.glob(os.path.join(args.data_dir, 'data*.h5')))
  error = np.array([0.0, 0.0, 0.0])
  count = 0
  for index in range(len(files)):
    print(files[index])
    data = h5py.File(files[index], 'r')
    for row in range(data['targets'].shape[0]):
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
        command = 3 # right
      else:
        assert(False)
      label = row_data[np.array([STEER, GAS, BRAKE])]

      image_input = img.astype(np.float32)
      image_input = np.multiply(image_input, 1.0 / 255.0)

      steer, gas, brake = agent._control_function(image_input, np.array([[speed]]), command, agent._sess)
      error[0] += get_mae(steer, row_data[0])
      error[1] += get_mae(gas, row_data[1])
      error[2] += get_mae(brake, row_data[2])
      count += 1

  print('mse error', error/count)
