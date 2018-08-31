Conditional Imitation Learning Training at CARLA
===============

Train deep learning model for conditional imitation learning using a modified neural network architecture from the orignal Carla paper. Model inputs include image, car speed and control input. The control can be 4 commands from Carla planner: follow lane, straight at intersection, left turn, right turn. The outputs of the model have 5 branches, one for each control input and the last one is to predict speed. Each of the first four branches has steering, gas, brake outputs. So the output is a 3 * 4 + 1 dimentional vector. The loss is MSE for the branch corresponding to the control input, plus speed. We use the same weighting proposed in the Carla paper:
```python
  lossWeights = {
    'steer': 0.95*0.45,
    'gas': 0.95*0.45,
    'brake': 0.95*0.05,
    "branch_speed": 0.05
  }
```

Training and validation data are provided by the authors of the Carla paper. We first obtain some statistics of the data. The input is sampled at 15fps, so the next game time is 67 millisecond apart within each episode. The game time between episodes are far apart. The x and y coordindates are in centimeters. The speed is in km/h. For example if speed is 50, the expected distance is 50e5 / 3600 / 15.0 = 92 centimeters. We also calculate the count and min max of the training data between different control commands. The data seems to be balanced. The training data also includes randomly injected triangle noise to demonstrate steering drift correction. During our training, no other transformations are performed except that images are agumented.

| control        | count           | steer  | gas  | brake  | speed  |
| ------------- | --- | ---------:|:---------:|:------------:| -----:|
| follow (2)      | 221757 | [-1.15, 1.18] | [0, 1] | [0, 1] | [-18, 82] |
| left (3)      | 131531 | [-1.02, 1.19] | [0, 1] | [0, 1] | [-18, 82] |
| right (4)      | 150624 | [-1.08, 1.19] | [0, 1] | [0, 1] | [-17, 82] |
| straight (5)      | 153356 | [-1., 1.15] | [0, 1] | [0, 1] | [-15, 82] |

The architecture roughly follows the paper, except we use a pretrained ResNet50 to extract image features. We first freeze the ResNet to train other layers using 0.01 learning rate for several epochs. Afterwards, the network is trained for 30 epochs using a decayed learning rate starting from 0.0001. The final training loss is 0.0081, validation loss is 0.0253. We also compared the MSE error on the validation dataset with the released official model.

| models        |  steer MSE  | gas MSE  | brake MSE  |
| ------------- |:-------------:|:-------------:| -----:|
| Official      | 0.02257433 | 0.07473299 | 0.04620963 |
| Ours      | 0.0197907 | 0.04334486 | 0.08165308 |

We also compare our model in Carla simulator for the CoRL17 benchmark. Only Town1 is tested (Town2's waypoint definition seem to have some mismatch, which prevent correct run). Percentage of successful episodes for test weathers unseen in training:

| models        |  Task 0  | Task 1  |  Task 2  | Task 3  |
| -------- |:----------:|:---------:|:-----------:| -----:|
| Official      | 0.98 | 0.94 | 0.82 | 0.74 |
| Ours      | 0.96 | 0.96 | 0.88 | 0.92 |

Big thanks to the Carla simulator and training data that make this work possible!

![alt tag](https://raw.githubusercontent.com/dongwang218/imitation-learning/mytraining/train/simulation.jpg)
