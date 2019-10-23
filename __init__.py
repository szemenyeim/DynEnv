from utils import GameType, ObservationType, NoiseType
from Control import Environment

class Env(object):
    def __init__(self,nPlayers,render=False,gameType = GameType.Full,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic):
        self.Internal = Environment(nPlayers,render,gameType,observationType,noiseType)

    def step(self,actions):
        return self.Internal.step(actions)

__all__ = ["Env","GameType","ObservationType","NoiseType",]