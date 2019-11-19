# DynEnv
Dynamic Simulation Environments for Reinforcement Learning

This project contains two reinforcement learning environments based on 2D physics simulation via [pymunk](https://www.pymunk.org). The environments support different observation modalities and also noisy observations. The current environments are the following:

- **Robot Soccer SPL League (RoboCupEnvironment):** Here, two teams of robots are competing to play soccer.
- **Autonomous driving environment (DrivingEnvironment):** Here, two teams of cars try to get to their unique destinations as quickly as possible without crashing or hitting pedestrians. The teams are not competing here, but only cars on the same team are allowed to share information (to model human drivers).

## Requirements

- Python 3.6+
- PyMunk
- OpenCV
- PyGame
- PyTorch (optional)

## Installation

You can install simply using pip:

`pip install dynenv`

Or build from source:

```
git clone https://github.com/szemenyeim/DynEnv.git
cd DynEnv
pip install -e .
```

## Usage

You can simply use the environments the following way:

```python
from DynEnv import *

myEnv = RoboCupEnvironment(nPlayers)
myEnv = DrivingEnvironment(nPlayers)

ret = myEnv.step(actions)
```

More complex examples can be found in the main.py and testNet.py files.

### Parameters

Here are some of the important settings of the environments

- **nPlayers [1-5]**: Number of total players per team. The total number of players is twice this in both environments.
- **render [bool]**: Whether to visualize the environment.
- **observationType [Full, Partial, Image]**: Image observation only supported for the RoboCup environment.
- **noiseType [Random, Realistic]**: Realistic noise: noise magnitude and false negative rate depends on distance, proximity of other objects and sighting type. False positives and misclassifications are more likely to occur in certain situations.
- **noiseMagnitude [0-5]**: Variable to control noise

Here are some examples of different noise and observation trypes

#### Random Noise

![Full](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/randNoise/game.gif)
![Partial](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/randNoise/obs.gif)
![Top camera image](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/randNoise/top.gif)
![Bottom camera image](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/randNoise/bottom.gif)

#### Realistic Noise

![Full](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/realNoise/game.gif)
![Partial](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/realNoise/obs.gif)
![Top camera image](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/realNoise/top.gif)
![Bottom camera image](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/realNoise/bottom.gif)

#### Large, Realistic Noise

![Full](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/bigNoise/game.gif)
![Partial](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/bigNoise/obs.gif)
![Top camera image](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/bigNoise/top.gif)
![Bottom camera image](https://raw.githubusercontent.com/szemenyeim/DynEnv/master/bigNoise/bottom.gif)

### Important functions

- `reset()` Resets the environment to a new game and returns initial observations.
- `setRandomSeed(seed)` Sets the environment seed, resets the environment and returns initial observations.
- `getObservationSize()` Returns information about the observations returned by the envionrment.
- `getActionSize()` Returns information about the actions the environment expects.
- `step(actions)` Performs one step. This consists of several simulation steps (10 for the Driving and 50 for the RoboCup envs). It returns observations for every 10 simulation steps and full state for the last step.

### So, what are the actions?

The environments expect an iterable object containing the actions for every player. Each player action must contain the following:

#### RoboCup:
- **Movement direction:** 0,1,2,3,4
- **Turn:** 0,1,2
- **Turn head:** between +/-6
- **Kick: 0,1,2** (this is exclusive with moving or turning)

#### Driving:
- **Gas/break:** between +/-3
- **Turn: between** +/- 3

### What is returned?

Both environments return the following variables in the step function:

- **Full state:** The full state variables.
- **Observations:** Observations for every robot/car. What this is exactly depends on the observationType variable.
- **Car/Robot rewards:** Rewards for each car or robot.
-- **Team rewards:** Shared rewards for every team. These are added to the Car/Robot rewards variables, and are not returned.
- **Finished:** Game over flag

Position information is normalized between +/-1 in both the observations and the full state.

#### RoboCup

The full state contains the following:

- Balls **[position, ball owned team ID]**
- Robots **[position, angle, team, fallen or penalized]**

If the observation is full state, the robot's own position is returned in a separate list, and the x axis is flipped for team 1. Moreover, in this case the ball owned flag indicates whether the ball is owned by the robot's team, or the opponent.

The partial observation contains the following for each robot:

- Balls: **[position, radius, ball owned status]**
- Robots (self not included): **[position, radius, angle, team, fallen or penalized]**
- Goalposts: **[position, radius]**
- Crosses: **[position, radius]**
- Lines: **[endpoint1, endpoint2]**
- Center circle: **[position, radius]**

sigthingType can be Normal, Distant or Partial. In this case, the positions and angles are returned relative to the robot's position and head angle.

The image observations contain 2D images of semantic segmentation data.

#### Driving

The full state contains the following:

- Cars: **[position, angle, width, height]**
- Obstacles: **[position,  angle, width, height]**
- Pedestrians: **[position]**
- Lanes: **[point1, point2, type]**

If the observation is full state, the car's own position is returned in a separate list.

The partial observation contains the following for each car:

- Self: **[position, angle, width, height, goal]**
- Cars: **[position, angle, width, height]**
- Obstacles: **[position, angle, width, height]**
- Pedestrians: **[position]**
- Lanes: **[point1, point2, type]**

Widths and heights are also normalized.
