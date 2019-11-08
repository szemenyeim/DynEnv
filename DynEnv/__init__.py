from .utils import ObservationType, NoiseType
from .RoboCupEnvironment import RoboCupEnvironment
from .DrivingEnvironment import DrivingEnvironment

class RoboCupEnv(object):
    def __init__(self,nPlayers,render=False,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic, noiseMagnitude = 2):
        self.Internal = RoboCupEnvironment(nPlayers,render,observationType,noiseType,noiseMagnitude)

    def step(self,actions):
        return self.Internal.step(actions)

class DrivingEnv(object):
    def __init__(self,nPlayers,render=False,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic, noiseMagnitude = 2):
        self.Internal = DrivingEnv(nPlayers,render,observationType,noiseType,noiseMagnitude)

    def step(self,actions):
        return self.Internal.step(actions)

__all__ = ["DrivingEnv","RoboCupEnv","ObservationType","NoiseType",]