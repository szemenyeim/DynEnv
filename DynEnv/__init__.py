from .utils import ObservationType, NoiseType, DynEnvType
from .RoboCupEnvironment import RoboCupEnvironment
from .DrivingEnvironment import DrivingEnvironment
from .CustomVecEnv import CustomSubprocVecEnv


""" Create dynamic environment with custom wrapper"""


def make_dyn_env(args):
    if args.env_type is DynEnvType.ROBO_CUP:
        envs = [lambda: RoboCupEnvironment(nPlayers=args.num_players, render=args.render, observationType=args.observationType,
                                                  noiseType=args.noiseType, noiseMagnitude=args.noiseMagnitude, allowHeadTurn=args.continuous)
                for i in range(args.num_envs)]
        env = CustomSubprocVecEnv(envs)
        name = "RoboCup"
    elif args.env_type is DynEnvType.DRIVE:
        envs = [lambda: DrivingEnvironment(nPlayers=args.num_players, render=args.render, observationType=args.observationType,
                                                  noiseType=args.noiseType, noiseMagnitude=args.noiseMagnitude, continuousActions=args.continuous)
                for i in range(args.num_envs)]
        env = CustomSubprocVecEnv(envs)
        name = "Driving"
    else:
        raise ValueError

    return env, name

__all__ = ["DrivingEnvironment","RoboCupEnvironment","ObservationType","NoiseType","make_dyn_env","DynEnvType"]