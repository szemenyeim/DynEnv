from .cutils import ObservationType, NoiseType, DynEnvType
from .RoboCupEnvironment import RoboCupEnvironment
from .DrivingEnvironment import DrivingEnvironment
from .utils.subproc_vec_env import SubprocVecEnv

def make_dyn_env(env, num_envs, num_players, render, observationType, noiseType, noiseMagnitude,
                 use_continuous_actions):
    if env is DynEnvType.ROBO_CUP:
        envs = [lambda: RoboCupEnvironment(nPlayers=num_players, render=render, observationType=observationType,
                                           noiseType=noiseType, noiseMagnitude=noiseMagnitude, obs_space_cast=True,
                                           allowHeadTurn=use_continuous_actions)
                for _ in range(num_envs)]
        env = SubprocVecEnv(envs)
        name = "RoboCup"
    elif env is DynEnvType.DRIVE:
        envs = [lambda: DrivingEnvironment(nPlayers=num_players, render=render, observationType=observationType,
                                           noiseType=noiseType, noiseMagnitude=noiseMagnitude, obs_space_cast=True,
                                           continuousActions=use_continuous_actions)
                for _ in range(num_envs)]
        env = SubprocVecEnv(envs)
        name = "Driving"
    else:
        raise ValueError

    return env, name

__all__ = ["DrivingEnvironment","RoboCupEnvironment","ObservationType","NoiseType","make_dyn_env","DynEnvType"]