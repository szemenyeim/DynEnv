import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunkoptions

pymunkoptions.options["debug"] = False
from DynEnv import *
from DynEnv.models.agent import ICMAgent
from DynEnv.examples.args import get_args
from DynEnv.models.train import Runner
from DynEnv.utils.utils import set_random_seeds, NetworkParameters, RewardType, AttentionTarget, AttentionType

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
    env, env_name = make_dyn_env(args.env, args.num_envs, args.num_players, args.render, args.observationType,
                                        args.noiseType, args.noiseMagnitude, args.use_continuous_actions)
    action_size = env.action_space
    obs_space = env.observation_space

    # True number of players: RoboCup Env asks for players per team
    num_players = args.num_players * 2 if args.env == DynEnvType.ROBO_CUP else args.num_players

    # agent
    agent = ICMAgent(args.num_envs, num_players, action_size, attn_target, attn_type, obs_space, feature_size,
                     args.forward_coeff, args.icm_beta, args.rollout_size, 5 if args.env is DynEnvType.ROBO_CUP else 1, lr=args.lr)

    # params
    param = NetworkParameters(env_name, args.num_envs, args.n_stack, args.rollout_size,
                              args.num_updates, args.max_grad_norm, args.icm_beta,
                              args.value_coeff, args.forward_coeff, args.entropy_coeff, attn_target, attn_type,
                              RewardType.INTRINSIC_AND_EXTRINSIC, args.note, args.use_full_entropy)

    # runner object & training
    runner = Runner(agent, env, param, args.cuda, args.seed, args.log_dir)
    runner.train()
