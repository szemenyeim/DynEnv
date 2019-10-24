from utils import ObservationType, NoiseType
from Control import Environment

class RoboEnv(object):
    def __init__(self,nPlayers,render=False,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic):
        self.Internal = Environment(nPlayers,render,observationType,noiseType)

    def step(self,actions):
        return self.Internal.step(actions)

__all__ = ["RoboEnv","ObservationType","NoiseType",]