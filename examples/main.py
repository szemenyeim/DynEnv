import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunkoptions
pymunkoptions.options["debug"] = False
import DynEnv
from .agent import ICMAgent
from .args import get_args
from .train import Runner
from .utils import set_random_seeds, NetworkParameters, RewardType, AttentionTarget, AttentionType
from .models import *


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
    env, env_name = DynEnv.make_dyn_env(DynEnv.DynEnvType.ROBO_CUP, args.num_players, args.num_envs)
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