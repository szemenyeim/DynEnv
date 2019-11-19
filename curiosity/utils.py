import os
import random
from enum import Enum
from os import makedirs, listdir
from os.path import isdir, isfile, join, dirname, abspath

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


class AttentionType(Enum):
    SINGLE_ATTENTION = 0
    DOUBLE_ATTENTION = 1


class AttentionTarget(Enum):
    NONE = 0
    ICM = 1
    A2C = 2
    ICM_LOSS = 3


class RewardType(Enum):
    INTRINSIC_AND_EXTRINSIC = 0
    INTRINSIC_ONLY = 1  # currently not used


def set_random_seeds(seed=42):
    """
    Courtesy of https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    :param seed:
    :return:
    """
    # 0. call stable baselines seed config routine
    from stable_baselines.common import set_global_seeds
    set_global_seeds(seed)

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.compat.v1.set_random_seed(seed)

    # 5. set the PyTorch seed + CUDA backend
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def label_converter(label):
    label = label[label.find(".") + 1:]

    if label == "NONE":
        label = "Baseline"
    elif label == "ICM_LOSS":
        label = "RCM"
    elif label == "SINGLE_ATTENTION":
        label = "single attention"
    elif label == "DOUBLE_ATTENTION":
        label = "double attention"

    return label

def instance2label(instance):
    # label generation
    label = f"{label_converter(series_indexer(instance['attention_target']))}, {label_converter(series_indexer(instance['attention_type']))}"
    # remove attention annotation from the baseline
    if "Baseline" in label:
        label = "Baseline"
    elif "RCM" in label:
        label = "RCM"
    elif "A2C" in label:
        label = "AttA2C"
    return label



def label_enum_converter(label):
    label = label[label.find(".") + 1:]

    if label == "NONE":
        label = AttentionTarget.NONE
    elif label == "ICM_LOSS":
        label = AttentionTarget.ICM_LOSS
    elif label == "SINGLE_ATTENTION":
        label = AttentionType.SINGLE_ATTENTION
    elif label == "DOUBLE_ATTENTION":
        label = AttentionType.DOUBLE_ATTENTION
    elif label == "A2C":
        label = AttentionTarget.A2C

    return label


def color4label(label):
    if label == "Baseline":
        color = "tab:blue"
    elif label == "AttA2C":
        color = "tab:purple"
    elif label == "ICM, single attention":
        color = "tab:red"
    elif label == "ICM, double attention":
        color = "tab:green"
    elif label == "RCM":
        color = "tab:orange"

    return color


def series_indexer(series):
    return series[series._index[0]]


def print_init(inset=True, zoom=2.5, loc=4):
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, tick_num: int(val*self.decimate_step)))
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=False)

    if loc == 1:
        bbox_to_anchor = (0.95, 0.95)
        loc1, loc2 = 3, 4
    elif loc == 2:
        bbox_to_anchor = (0.08, .95)
        loc1, loc2 = 1, 3
    elif loc == 4:
        bbox_to_anchor = (0.95, .1)
        loc1, loc2 = 2, 4

    if inset:
        axins = zoomed_inset_axes(ax, zoom=zoom, loc=loc, bbox_to_anchor=bbox_to_anchor,
                                  bbox_transform=ax.transAxes)  # zoom-factor: 2, location: upper-left
        axins.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=False)
    else:
        axins = None

    return fig, ax, axins, loc1, loc2


def plot_postprocess(fig, ax, keyword, env, dir, xlabel="Rollout", save=False):
    # assemble notation
    if keyword == "rewards":
        stat_descriptor = r"$\mu_{reward}$"
        file_prefix = "mean_reward"
    elif keyword == "features":
        stat_descriptor = r"$\sigma_{feature}$"
        file_prefix = "feat_std"

    title = stat_descriptor + f" in {env}"

    # set plot descriptors
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(stat_descriptor)

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.125),
                       fancybox=True, shadow=False, ncol=2)

    # legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125),
    #                    fancybox=True, shadow=False, ncol=2)

    if save:
        fig.savefig(join(dir, f"{file_prefix}_{env}.svg"), bbox_extra_artists=(legend,), bbox_inches='tight',
                    format="svg", transparent=True)
        fig.savefig(join(dir, f"{file_prefix}_{env}.png"), bbox_extra_artists=(legend,), bbox_inches='tight',
                    format="png", transparent=True)


class HyperparamScheduler(object):

    def __init__(self, init_val, end_val=0.0, tau=20000, threshold=1e-5):
        super().__init__()

        self.init_val = init_val
        self.end_val = end_val
        self.value = self.init_val
        self.cntr = 0
        self.tau = tau
        self.threshold = threshold

    def step(self):
        self.cntr += 1

        if self.value > self.threshold:
            self.value = self.end_val + (self.init_val - self.end_val) * np.exp(-self.cntr / self.tau)
        else:
            self.value = 0.0

    def save(self, group):
        """

        :param group: the reference to the group level hierarchy of a .hdf5 file to save the data
        :return:
        """
        for key, val in self.__dict__.items():
            group.create_dataset(key, data=val)


class NetworkParameters(object):
    def __init__(self, env_name: str, num_envs: int, n_stack: int, rollout_size: int = 5, num_updates: int = 2500000,
                 max_grad_norm: float = 0.5,
                 icm_beta: float = 0.2, value_coeff: float = 0.5, entropy_coeff: float = 0.02,
                 attention_target: AttentionTarget = AttentionTarget.NONE,
                 attention_type: AttentionType = AttentionType.SINGLE_ATTENTION,
                 reward_type: RewardType = RewardType.INTRINSIC_ONLY):
        self.env_name = env_name
        self.num_envs = num_envs
        self.n_stack = n_stack
        self.rollout_size = rollout_size
        self.num_updates = num_updates
        self.max_grad_norm = max_grad_norm
        self.icm_beta = icm_beta
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.attention_target = attention_target
        self.attention_type = attention_type
        self.reward_type = reward_type

    def save(self, data_dir, timestamp):
        param_dict = {**self.__dict__, "timestamp": timestamp}

        df_path = join(data_dir, "params.tsv")

        pd.DataFrame.from_records([param_dict]).to_csv(
            df_path,
            sep='\t',
            index=False,
            header=True if not isfile(df_path) else False,
            mode='a')


def make_dir(dirname):
    if not isdir(dirname):
        makedirs(dirname)


def load_and_eval(agent, env, steps=2500):
    agent.load_state_dict(torch.load("best_agent"))
    agent.eval()

    images = []
    obs = env.reset()
    for _ in range(steps):
        tensor = torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.
        tensor = tensor.cuda() if torch.cuda.is_available() else tensor
        action, _, _, _, _ = agent.a2c.get_action(tensor)
        _, _, _, _ = env.step(action)
        images.append(env.render(mode="rgb_array"))

    imageio.mimsave('lander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)


def merge_tables():
    # iterate over tables
    log_dir = join(dirname(dirname(abspath(__file__))), "log")

    for env_dir in listdir(log_dir):
        stocks = []
        data_dir = join(log_dir, env_dir)
        for table in listdir(data_dir):
            if table.endswith(".tsv"):
                stock_df = pd.read_csv(join(data_dir, table), sep="\t")
                stocks.append(stock_df)
        pd.concat(stocks, axis=0, sort=True).to_csv(join(data_dir, "params.tsv"), sep="\t", index=False)


def numpy_ewma_vectorized_v2(data, window):
    """
    Source: https://stackoverflow.com/a/42926270
    :param data:
    :param window:
    :return:
    """

    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


class AgentCheckpointer(object):

    def __init__(self, env_name, num_updates, timestamp, log_dir=None, log_points=(.25, .5, .75, .99)) -> None:
        super().__init__()

        # constants
        self.timestamp = timestamp
        self.num_updates = num_updates
        self.update_cntr = 0
        self.best_loss = np.inf
        self.best_reward = -np.inf
        log_keys = np.int32(self.num_updates * np.array(log_points)).tolist()
        self.log_points = dict(zip(log_keys, log_points))

        # file structure
        self.base_dir = join(dirname(dirname(abspath(__file__))), "log") if log_dir is None else log_dir
        self.data_dir = join(self.base_dir, env_name)
        make_dir(self.base_dir)
        make_dir(self.data_dir)

    def checkpoint(self, loss, reward, agent):
        mean_reward = np.array(reward).mean()

        # save agent with lowest loss
        if loss < self.best_loss:
            self.best_loss = loss.item()
            torch.save(agent.state_dict(), join(self.data_dir, f"agent_best_loss_{self.timestamp}"))

        # save agent with highest mean reward
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            torch.save(agent.state_dict(), join(self.data_dir, f"agent_best_reward_{self.timestamp}"))

        # save agent at specific time intervals
        if self.update_cntr in self.log_points.keys():
            torch.save(agent.state_dict(),
                       join(self.data_dir, f"agent_step_{self.log_points[self.update_cntr]}_{self.timestamp}"))

        self.update_cntr += 1


from matplotlib import rc


def plot_typography(usetex=True, small=12, medium=14, big=16):
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:

    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=usetex)
    rc('font', family='serif')


    rc('font', size=small)  # controls default text sizes
    rc('axes', titlesize=small)  # fontsize of the axes title
    rc('axes', labelsize=medium)  # fontsize of the x and y labels
    rc('xtick', labelsize=small)  # fontsize of the tick labels
    rc('ytick', labelsize=small)  # fontsize of the tick labels
    rc('legend', fontsize=small)  # legend fontsize
    rc('figure', titlesize=big)  # fontsize of the figure title
