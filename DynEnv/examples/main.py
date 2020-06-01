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
    feature_size = 64
    attn_target = AttentionTarget.ICM_LOSS
    attn_type = AttentionType.SINGLE_ATTENTION

    # env
    env, env_name = make_dyn_env(args.env, args.num_envs, args.num_players, args.render, args.observationType,
                                 args.noiseType, args.noiseMagnitude, args.use_continuous_actions)
    action_size = env.action_space
    obs_space = env.observation_space

    # True number of players: RoboCup Env asks for players per team
    num_players = args.num_players * 2 if args.env == DynEnvType.ROBO_CUP else args.num_players

    reco_desc = env.get_attr('recoDescriptor', 0)[0]

    # agent
    agent = ICMAgent(args, num_players, action_size, attn_target, attn_type, obs_space, feature_size, reco_desc,
                     5 if args.env is DynEnvType.ROBO_CUP else 1, 6 if args.env is DynEnvType.ROBO_CUP else 4)

    # params
    param = NetworkParameters(env_name, args, attn_target, attn_type, RewardType.INTRINSIC_AND_EXTRINSIC)

    # runner object & training
    runner = Runner(agent, env, param, args)
    runner.train()
