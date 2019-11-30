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

`pip install DynEnv`

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

Or create vectorized environments by using:

```python
make_dyn_env(env, num_envs, num_players, render, observationType, noiseType, noiseMagnitude, use_continuous_actions)
env = DynEnvType.ROBO_CUP or DynEnvType.DRIVE
```

More complex examples including 
- neural networks tailored for the special output format (i.e. the number of observations can vary through time),
- logging and
- plotting the results.

For the above, confer the `DynEnv/examples` directory. The `main.py` file consists a full example, while if you would like to try out how the environments work by hand, `play.py` is there for you as well.

### Model structure
The most important part from the point of view of the neural network is the `DynEnv/models` directory, which exposes you the following classes:
- _ICMAgent_: the top-level agent consisting of an A2C and an Intrinsic Curiosity Module (and its variant, [Rational Curiosity Module](https://github.com/rpatrik96/AttA2C))
- _InOutArranger_: helper class to rearrange observations for simple NN forwarding
- _EmbedBlock_: the embedding network used for an object
- _InputLayer_: a complex network which convert all observations into a unified feature space
- _ActorBlock_: a neural network predicting actions for a given action type
- _ActorLayer_: an ensemble of _ActorBlock_ to predict every action
- _AttentionLayer_:
- _DynEnvFeatureExtractor_: a wrapper for the input transform by _InputLayer_, collapsing the time dimension with Recurrent Temporal Attention and running an LSTM


### Parameters

Here are some of the important settings of the environments

- **nPlayers [1-5/10]**: Number of total players in the environment (in the RoboCup env this is per team). The limit is 10 in the Driving, 5 in the RoboCup env.
- **render [bool]**: Whether to visualize the environment.
- **observationType [Full, Partial, Image]**: Image observation only supported for the RoboCup environment.
- **noiseType [Random, Realistic]**: Realistic noise: noise magnitude and false negative rate depends on distance, proximity of other objects and sighting type. False positives and misclassifications are more likely to occur in certain situations.
- **noiseMagnitude [0-5]**: Variable to control noise
- **continuousActions [bool]**: Whether the driving env actions are understood as categorical or continuous. (Driving env only)
- **allowHeadturn [bool]**: Enables head turining actions. (RoboCup env only)

Here are some examples of different noise and observation types

#### Random Noise

Full Observation            |  Partial Observation
:-------------------------:|:-------------------------:
![](/images/randNoise/game.gif)  |  ![](/images/randNoise/obs.gif)

Top Camera            |  Bottom Camera
:-------------------------:|:-------------------------:
![](/images/randNoise/top.gif)  |  ![](/images/randNoise/bottom.gif)


#### Realistic Noise


Full Observation            |  Partial Observation
:-------------------------:|:-------------------------:
![](/images/realNoise/game.gif)  |  ![](/images/realNoise/obs.gif)

Top Camera            |  Bottom Camera
:-------------------------:|:-------------------------:
![](/images/realNoise/top.gif)  |  ![](/images/realNoise/bottom.gif)

#### Large, Realistic Noise


Full Observation            |  Partial Observation
:-------------------------:|:-------------------------:
![](/images/bigNoise/game.gif)  |  ![](/images/bigNoise/obs.gif)

Top Camera            |  Bottom Camera
:-------------------------:|:-------------------------:
![](/images/bigNoise/top.gif)  |  ![](/images/bigNoise/bottom.gif)

#### Driving, Realistic Noise


Full Observation            |  Partial Observation
:-------------------------:|:-------------------------:
![](/images/drive/game.gif)  |  ![](/images/drive/obs.gif)

### Important functions and members

- `reset()` Resets the environment to a new game and returns initial observations.
- `setRandomSeed(seed)` Sets the environment seed, resets the environment and returns initial observations.
- `observationSpace` Returns information about the observations returned by the environrment. (**Note: These are not Gym Compatible, see the folowing section**)
- `actionSpace` Returns information about the actions the environment expects.
- `step(actions)` Performs one step. This consists of several simulation steps (10 for the Driving and 50 for the RoboCup environments). It returns observations for every 10 simulation steps and full state for the last step.
- `renderMode` Whether to render to a display (`'human'`) or to a memory array (`'memory'`).
- `agentVisID` With this, you can visualize the observation of an agent during rendering.
- `render()` Returns rendered images if the render mode is `'memory'`. Does nothing otherwise, as the rendering is done by the step function due to the multi-timestep feature.

### So, what are the actions?

The environments expect an iterable object containing the actions for every player. Each player action must contain the following:

#### RoboCup:
- **Movement direction:** Categorical (5)
- **Turn:** Categorical (3)
- **Turn head:** Continuous [-6 +6]
- **Kick:** Categorical (3) (this is exclusive with moving or turning)

#### Driving:
- **Gas/break:** Continuous [-3 +3] or Categorical (3)
- **Turn:** Continuous [-3 +3] or Categorical (3)

### What is returned?

Both environments return the following variables in the step function:

- **Observations:** Observations for every agent. What this is exactly depends on the observationType variable.
- **Rewards:** Rewards for each agent.
  - **Team rewards:** Shared rewards for every team. These are added to the agent reward variables, and are not returned.
- **Finished:** Game over flag
- **Info:** Other important data
  - **Full State:** The full state of the env
  - **episode_r**: Cumulative rewards for the episode (Returned only at the end of an episode)
  - **episode_p_r**: Cumulative positive-only rewards for the episode (Returned only at the end of an episode)
  - **episode_g**: Goals in these episode (RoboCup env only) (Returned only at the end of an episode)

Position information is normalized in both the observations and the full state.

#### The Observation Space

Due to limitations in the OpenAI gym, this part of the env is not fully compatible. The `observationSpace` variable is a list containing the following variables:

`[nTimeSteps, nAgents, nObjectType, [<number of features for every object type>]]`
 
The observations returned are arranged as follows:
 
`[nParallelEnvs x nTimeSteps x nAgents x nObjectType]`
  
Each element of the above list is a NumPy array containing all the observations by a single agent in a single timestep. To help contructing input layers a custom class `DynEnv.models.InOutArranger` is provided with the following two functions:

- `inputs, counts = rearrange_inputs(x)`: Creates a single list of NumPy arrays. Each element of this list contains a single numpy array of all the observations for a given object type. (Warning: in some cases this might be an empty list!)
- `outputs, masks = rearrange_outputs(inputs, counts)`: Takes a list of Torch Tensors and the counts output by the previous function, and creates a single tensor shaped [TimeSteps x maxObjCnt x nPlayers x featureCnt] by padding the second dimension to the largest number of objects seen for every robot. The masks variable is binary array shaped [TimeSteps x maxObjCnt x nPlayers], which is True for padded elements (this is in line with PyTorch's MultiHeadedAttention layer). (Warning: This assumes that the featureCnt is the same for every object time.)

Here is a more comprehensive example:

```python
from DynEnv.models import *
from torch import nn

myEnv = ...
obsSpace = myEnv.observationSpace
nTime = obsSpace[0]
nPlayers = obsSpace[1]
nObj = obsSpace[2]
featuresPerObject = obsSpace[3]

device = <CUDA or CPU>
myNeuralNets = [nn.Linear(objfeat,128).to(device) for objFeat in featuresPerObject]
myArranger = models.InOutArranger(nObj,nPlayers,nTime)

...

obs, _ = myEnv.step(actions)

netInputs, counts = myArranger.rearrange_inputs(obs)
netOutputs = [myNet(torch.tensor(netInput).to(device)) for myNet,netInput in zip(myNeuralNets,netInputs)]
outputs,masks = myArranger.rearrange_outputs(netOutputs,counts)

```

#### RoboCup

The full state contains the following:

- Robots **[x, y, cos(angle), sin(angle), team, fallen or penalized]**
- Balls **[x, y, ball owned team ID]**

If the observation is full state, the robot's own position is returned in a separate list, and both axes are flipped and angles rotated 180 degrees for team -1. Moreover, in this case the ball owned flag indicates whether the ball is owned by the robot's team, or the opponent.

The partial observation contains the following for each robot:

- Balls: **[x, y, radius, ball owned status]**
- Robots (self not included): **[x, y, radius, cos(angle), sin(angle), team, fallen or penalized]**
- Goalposts: **[x, y, radius]**
- Crosses: **[x, y, radius]**
- Lines: **[x1, y1, x2, y2]**
- Center circle: **[x, y, radius]**

sigthingType can be Normal, Distant or Partial. In this case, the positions and angles are returned relative to the robot's position and head angle.

The image observations contain 2D images of semantic labels. The images have 4 binary channels:

- 0: Ball
- 1: Robot
- 2: Goalpost
- 3: Line

#### Driving

The full state contains the following:

- Cars: **[x, y, cos(angle), sin(angle), width, height, finished]**
- Obstacles: **[x, y, cos(angle), sin(angle), width, height]**
- Pedestrians: **[x, y]**
- Lanes: **[x1, y1, x2, y2, type]**

If the observation is full state, the car's own position is returned in a separate list, identical to the Self entry below.

The partial observation contains the following for each car:

- Self: **[x, y, cos(angle), sin(angle), width, height, goal_x, goal_y, finished]**
- Cars: **[x, y, cos(angle), sin(angle), width, height]**
- Obstacles: **[x, y, cos(angle), sin(angle), width, height]**
- Pedestrians: **[x, y]**
- Lanes: **[signed distance, cos(angle), sin(angle), type]**

Widths and heights are also normalized.
