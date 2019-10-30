from .utils import ObservationType, NoiseType
from .Control import Environment

class RoboEnv(object):
    def __init__(self,nPlayers,render=False,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic, noiseMagnitude = 2):
        self.Internal = Environment(nPlayers,render,observationType,noiseType,noiseMagnitude)

    def step(self,actions):
        return self.Internal.step(actions)

__all__ = ["RoboEnv","ObservationType","NoiseType",]