from enum import Enum
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunkoptions
pymunkoptions.options["debug"] = False
import DynEnv
from curiosity.agent import ICMAgent
from curiosity.args import get_args
from curiosity.train import Runner
from curiosity.utils import set_random_seeds, NetworkParameters, RewardType, AttentionTarget, AttentionType
from models import *


class DynEnvType(Enum):
    ROBO_CUP = 0
    DRIVE = 1


def env_selector(env_type: DynEnvType, nPlayers, nEnvs):
    if env_type is DynEnvType.ROBO_CUP:
        envs = [lambda: DynEnv.RoboCupEnvironment(nPlayers=nPlayers, render=False, observationType=DynEnv.ObservationType.Full,
                                        noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=0.0) for i in range(nEnvs)]
        env = DynEnv.CustomSubprocVecEnv(envs)
        name = "RoboCup"
    elif env_type is DynEnvType.DRIVE:
        envs = [lambda: DynEnv.DrivingEnvironment(nPlayers=nPlayers, render=False, observationType=DynEnv.ObservationType.Partial,
                                        noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=0.1) for i in range(nEnvs)]
        env = DynEnv.CustomSubprocVecEnv(envs)
        name = "Driving"
    else:
        raise ValueError

    return env, name


if __name__ == '__main__':
    # arg parsing
    args = get_args()

    # seeds set (all but env)
    set_random_seeds(args.seed)

    # constants
    feature_size = 128
    attn_target = AttentionTarget.ICM_LOSS
    attn_type = AttentionType.SINGLE_ATTENTION

    # env
    env, env_name = env_selector(DynEnvType.ROBO_CUP, args.num_players, args.num_envs)
    batch, action_size = env.action_space
    input_size = env.observation_space

    # agent
    agent = ICMAgent(args.num_envs, args.num_players, action_size, attn_target, attn_type,
                     input_size, feature_size, args.forward_coeff, args.icm_beta, args.rollout_size, lr=args.lr)

    # params
    param = NetworkParameters(env_name, args.num_envs, args.n_stack, args.rollout_size,
                              args.num_updates, args.max_grad_norm, args.icm_beta,
                              args.value_coeff, args.forward_coeff, args.entropy_coeff, attn_target, attn_type,
                              RewardType.INTRINSIC_AND_EXTRINSIC, args.note, args.use_full_entropy)

    # runner object & training
    runner = Runner(agent, env, param, args.cuda, args.seed, args.log_dir)
    runner.train()
