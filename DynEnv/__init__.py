from .utils import ObservationType, NoiseType
from .RoboCupEnvironment import RoboCupEnvironment
from .DrivingEnvironment import DrivingEnvironment
from .CustomVecEnv import CustomSubprocVecEnv
from enum import Enum


class DynEnvType(Enum):
    ROBO_CUP = 0
    DRIVE = 1


""" Create dynamic environment with custom wrapper"""


def make_dyn_env(env_type: DynEnvType, nPlayers, nEnvs, render=False, observationType=ObservationType.Partial,
                 noiseType=NoiseType.Realistic, noiseMagnitude=0.1):
    if env_type is DynEnvType.ROBO_CUP:
        envs = [lambda: RoboCupEnvironment(nPlayers=nPlayers, render=render, observationType=observationType,
                                                  noiseType=noiseType, noiseMagnitude=noiseMagnitude) for i in
                range(nEnvs)]
        env = CustomSubprocVecEnv(envs)
        name = "RoboCup"
    elif env_type is DynEnvType.DRIVE:
        envs = [lambda: DrivingEnvironment(nPlayers=nPlayers, render=render, observationType=observationType,
                                                  noiseType=noiseType, noiseMagnitude=noiseMagnitude) for i in
                range(nEnvs)]
        env = CustomSubprocVecEnv(envs)
        name = "Driving"
    else:
        raise ValueError

    return env, name

__all__ = ["DrivingEnvironment","RoboCupEnvironment","ObservationType","NoiseType","make_dyn_env","DynEnvType"]