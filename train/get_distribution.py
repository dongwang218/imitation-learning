import os, sys
import h5py
import numpy as np

# if game time advance by 67, it is same sequence,
def speed_location(data):
  speed = data['targets'][:, 10][:-1] * 1000 / 3600 * 100 / 1000.0 # centimeter / millisecond
  delta_time = data['targets'][1:, 20] - data['targets'][:-1, 20]
  distance = speed * delta_time
  motion = np.linalg.norm(data['targets'][1:, 8:10] - data['targets'][:-1, 8:10], axis=-1)
  return np.abs(motion - distance)

COMMAND=24
MOTION=160 # centimeter
GAME_TIME = 67
indexes = np.array([0, 1, 2, 10])
# for each command, collect the minmax of each control
command = {}
prev_data = None
prev_f = None

for index, f in enumerate(sys.argv[1:]):
  print(f)
  sys.stdout.flush()

  data = h5py.File(f, 'r')
  if prev_data is not None:
    # most sequences are next to each other
    motion = np.linalg.norm(data['targets'][0, 8:10] - prev_data['targets'][-1, 8:10], axis = -1)
    if motion > MOTION and np.abs(data['targets'][0, 20] - prev_data['targets'][-1, 20]) <= GAME_TIME*2:
      print('%s leaped from last %s by %s' % (f, prev_f, motion))

    motion = np.linalg.norm(data['targets'][1:, 8:10] - data['targets'][:-1, 8:10], axis=-1)
    delta_time = np.abs(data['targets'][1:, 20] - data['targets'][:-1, 20])
    motion = motion[delta_time <= GAME_TIME]
    if np.any(motion > MOTION):
      print('%s leaped inside by %s' % (f, np.max(motion)))

    #gametime = np.max(delta_time)
    #if gametime > GAME_TIME:
    #  print('%s leaped game time inside by %s, median %s' % (f, gametime, np.median(delta_time)))

  prev_data = data
  prev_f = f

  for row in data['targets']:
    c = int(row[COMMAND])
    if c not in command:
      command[c] = {'count': 1, 'min': row[indexes], 'max': row[indexes]}
    else:
      command[c]['count'] += 1
      command[c]['min'] = np.minimum(command[c]['min'], row[indexes])
      command[c]['max'] = np.maximum(command[c]['max'], row[indexes])

    if command[c]['min'][-1] < -10:
      pass

  if index % 100 == 0:
    print(command)

  continue

print('final')
print(command)

# {0: {'count': 16, 'max': array([ 0.05718994,  0.9173279 ,  0.        , 45.216064  ], dtype=float32), 'min': array([-3.1219482e-02,  1.5258789e-05,  0.0000000e+00,  3.0974443e+01],                                         dtype=float32)}, 2: {'count': 23762, 'max': array([ 0.84704536,  1.        ,  1.        , 82.52298   ], dtype=float32), 'min': array([-0.96487176,  0.        ,  0.        , -4.780551  ], dtype=float32)}, 3: {'count': 15971, 'max': array([ 0.7377314,  1.       ,  1.       , 82.66553  ], dtype=float32), 'min': array([-1.       ,  0.       ,  0.       , -2.2824152], dtype=float32)}, 4: {'count': 15550, 'max': array([ 1.058575,  1.      ,  1.      , 81.99749 ], dtype=float32), 'min': array([-1.      ,  0.      ,  0.      , -9.223941], dtype=float32)}, 5: {'count': 18901, 'max': array([ 1.1992229,  1.       ,  1.       , 81.75     ], dtype=float32), 'min': array([ -1.      ,   0.      ,   0.      , -15.157239], dtype=float32)}}

#{0: {'count': 332, 'max': array([ 0.42355347,  1.        ,  0.        , 70.78535   ], dtype=float32), 'min': array([-0.18734741,  0.        ,  0.        ,  0.        ], dtype=float32)}, 2: {'count': 221757, 'max': array([ 1.1846792,  1.       ,  1.       , 82.72486  ], dtype=float32), 'min': array([ -1.1543294,   0.       ,   0.       , -18.280296 ], dtype=float32)}, 3: {'count': 131531, 'max': array([ 1.1989188,  1.       ,  1.       , 82.366516 ], dtype=float32), 'min': array([ -1.0205905,   0.       ,   0.       , -18.739027 ], dtype=float32)}, 4: {'count': 150624, 'max': array([ 1.1970137,  1.       ,  1.       , 82.635796 ], dtype=float32), 'min': array([ -1.084527,   0.      ,   0.      , -17.928936], dtype=float32)}, 5: {'count': 153356, 'max': array([ 1.1598685,  1.       ,  1.       , 82.72942  ], dtype=float32), 'min': array([ -1.0031055,   0.       ,   0.       , -15.7844305], dtype=float32)}}
