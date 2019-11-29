import argparse


def get_args():
    """
    Function for handling command line arguments

    :return: parsed   command line arguments
    """
    parser = argparse.ArgumentParser(description='Curiosity-driven deep RL with A2C+ICM')

    # training
    parser.add_argument('--train', action='store_true', default=False,
                        help='train flag (False->load model)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='CUDA flag')
    parser.add_argument('--log-dir', type=str, default="./log",
                        help='log directory')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed')
    parser.add_argument('--max-grad_norm', type=float, default=.5, metavar='MAX_GRAD_NORM',
                        help='threshold for gradient clipping')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')

    # environment
    parser.add_argument('--idx', type=int, default=9, metavar='IDX',
                        help='index of the configuration to start from (inclusive)')
    parser.add_argument('--num-train', type=int, default=5, metavar='NUM_TRAIN',
                        help='number of trainings to run')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4',
                        help='environment name')
    parser.add_argument('--num-envs', type=int, default=8, metavar='NUM_ENVS',
                        help='number of parallel environments')
    parser.add_argument('--num-players', type=int, default=2, metavar='NUM_PLAYERS',
                        help='number of players in the environment')
    parser.add_argument('--n-stack', type=int, default=1, metavar='N_STACK',
                        help='number of frames stacked')
    parser.add_argument('--rollout-size', type=int, default=10, metavar='ROLLOUT_SIZE',
                        help='rollout size')
    parser.add_argument('--num-updates', type=int, default=12000, metavar='NUM_UPDATES',
                        help='number of updates')
    parser.add_argument('--use-full-entropy', type=bool, default=False, metavar='USE_FULL_ENTROPY',
                        help='use full entropy, not just batch entropy')


    # model coefficients
    parser.add_argument('--icm-beta', type=float, default=1e-2, metavar='ICM_BETA',
                        help='beta for the ICM module')
    parser.add_argument('--value-coeff', type=float, default=5e-1, metavar='VALUE_COEFF',
                        help='value loss weight factor in the A2C loss')
    parser.add_argument('--entropy-coeff', type=float, default=1e-1, metavar='ENTROPY_COEFF',
                        help='entropy loss weight factor in the A2C loss')
    parser.add_argument('--forward-coeff', type=float, default=1e-2, metavar='FORWARD_COEFF',
                        help='forward loss weight factor in the ICM loss')

    parser.add_argument('--note', type=str, default='None',
                        help='Add note to help identify run')

    # Argument parsing
    return parser.parse_args()