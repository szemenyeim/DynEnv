from enum import Enum

import DynEnv
from curiosity.agent import ICMAgent
from curiosity.args import get_args
from curiosity.train import Runner
from curiosity.utils import set_random_seeds, NetworkParameters, RewardType, AttentionTarget, AttentionType
from models import *


class DynEnvType(Enum):
    ROBO_CUP = 0
    DRIVE = 1


def env_selector(env_type: DynEnvType, nPlayers):
    if env_type is DynEnvType.ROBO_CUP:
        env = DynEnv.RoboCupEnvironment(nPlayers=nPlayers, render=False, observationType=DynEnv.ObservationType.Partial,
                                        noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=0.1)
        name = "RoboCup"
    elif env_type is DynEnvType.DRIVE:
        env = DynEnv.DrivingEnvironment(nPlayers=nPlayers, render=False, observationType=DynEnv.ObservationType.Partial,
                                        noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=0.1)
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
    num_players = 1
    num_env = 1
    feature_size = 128
    attn_target = AttentionTarget.NONE
    attn_type = AttentionType.SINGLE_ATTENTION

    # env
    env, env_name = env_selector(DynEnvType.ROBO_CUP, num_players)
    batch, action_size = env.getActionSize()
    input_size = env.getObservationSize()

    # agent
    agent = ICMAgent(args.n_stack, args.num_envs, num_players, action_size, attn_target, attn_type,
                     input_size, feature_size, lr=args.lr)

    # params
    param = NetworkParameters(env_name, args.num_envs, args.n_stack, args.rollout_size,
                              args.num_updates, args.max_grad_norm, args.icm_beta,
                              args.value_coeff, args.entropy_coeff, attn_target, attn_type,
                              RewardType.INTRINSIC_AND_EXTRINSIC)

    # runner object & training
    runner = Runner(agent, env, param, args.cuda, args.seed, args.log_dir)
    runner.train()
