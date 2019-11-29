from os.path import abspath, dirname, join

import h5py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import make_dir, plot_postprocess, print_init, color4label, instance2label


class LogData(object):
    def __init__(self):
        self.values = []
        self.mean = []
        self.std = []
        self.min = []
        self.max = []

    def log(self, sample):
        """

        :param sample: data for logging specified as a numpy.array
        :return:
        """
        self.values.append(sample.mean(axis=1))
        self.mean.append(sample.mean())
        self.std.append(sample.std())
        self.min.append(sample.min())
        self.max.append(sample.max())

    def save(self, group):
        """

        :param group: the reference to the group level hierarchy of a .hdf5 file to save the data
        :return:
        """
        for key, val in self.__dict__.items():
            group.create_dataset(key, data=val)

    def load(self, group, decimate_step=100):
        """
        :param decimate_step:
        :param group: the reference to the group level hierarchy of a .hdf5 file to load
        :return:
        """
        # read in parameters
        # [()] is needed to read in the whole array if you don't do that,
        #  it doesn't read the whole data but instead gives you lazy access to sub-parts
        #  (very useful when the array is huge but you only need a small part of it).
        # https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
        self.values = group["values"][()][::decimate_step]
        self.mean = group["mean"][()][::decimate_step]
        self.std = group["std"][()][::decimate_step]
        self.min = group["min"][()][::decimate_step]
        self.max = group["max"][()][::decimate_step]

    def plot_mean_min_max(self, label):
        plt.fill_between(range(len(self.mean)), self.max, self.min, alpha=.5)
        plt.plot(self.mean, label=label)

    def plot_mean_std(self, label):
        mean = np.array(self.mean)
        plt.fill_between(range(len(self.mean)), mean + self.std, mean - self.std, alpha=.5)
        plt.plot(self.mean, label=label)


class TemporalLogger(object):
    def __init__(self, env_name, timestamp, log_dir, *args):
        """
        Creates a TemporalLogger object. If the folder structure is nonexistent, it will also be created
        :param *args:
        :param env_name: name of the environment
        :param timestamp: timestamp as a string
        :param log_dir: logging directory, if it is None, then logging will be at the same hierarchy level as src/
        """
        super().__init__()
        self.timestamp = timestamp

        # file structure
        self.base_dir = join(dirname(dirname(abspath(__file__))), "log") if log_dir is None else log_dir
        self.data_dir = join(self.base_dir, env_name)
        make_dir(self.base_dir)
        make_dir(self.data_dir)

        # data
        for data in args:
            self.__dict__[data] = LogData()

    def log(self, **kwargs):
        """
        Function for storing the new values of the given attribute
        :param **kwargs:
        :return:
        """
        for key, value in kwargs.items():
            self.__dict__[key].log(value)

    def save(self, *args):
        """
        Saves the temporal statistics into a .hdf5 file
        :param **kwargs:
        :return:
        """
        with h5py.File(join(self.data_dir, 'time_log_' + self.timestamp + '.hdf5'), 'w') as f:
            for arg in args:
                self.__dict__[arg].save(f.create_group(arg))

    def load(self, filename, decimate_step=100):
        """
        Loads the temporal statistics and fills the attributes of the class
        :param decimate_step:
        :param filename: name of the .hdf5 file to load
        :return:
        """
        if not filename.endswith('.hdf5'):
            filename = filename + '.hdf5'

        with h5py.File(join(self.data_dir, filename), 'r') as f:
            for key, value in self.__dict__.items():
                if isinstance(value, LogData):
                    value.load(f[key], decimate_step)

    def plot_mean_min_max(self, *args):
        fig, ax, _ = print_init(False)
        for arg in args:
            # breakpoint()
            if arg in self.__dict__.keys():  # and isinstance(self.__dict__[arg], LogData):
                self.__dict__[arg].plot_mean_min_max(arg)
        plt.title("Mean and min-max statistics")

    def plot_mean_std(self, *args):
        fig, ax, _ = print_init(False)
        for arg in args:
            if arg in self.__dict__.keys():
                self.__dict__[arg].plot_mean_std(arg)

        plt.title("Mean and standard deviation statistics")


class EnvLogger(object):

    def __init__(self, env_name, log_dir, decimate_step=250) -> None:
        super().__init__()
        self.env_name = env_name
        self.log_dir = log_dir
        self.decimate_step = decimate_step
        self.data_dir = join(self.log_dir, self.env_name)
        self.fig_dir = self.base_dir = join(dirname(dirname(abspath(__file__))), join("figures", self.env_name))
        make_dir(self.fig_dir)

        self.params_df = pd.read_csv(join(self.data_dir, "params.tsv"), "\t")

        self.logs = {}

        mean_ep_reward = []
        mean_ep_pos_reward = []

        # load trainings
        for timestamp in self.params_df.timestamp:

            # collect features to load
            features = ["ep_rewards"]
            if "Robo" in self.env_name:
                features += ["ep_pos_rewards"]

            self.logs[timestamp] = TemporalLogger(self.env_name, timestamp, self.log_dir, *features)
            self.logs[timestamp].load(join(self.data_dir, f"time_log_{timestamp}"), self.decimate_step)

            # calculate statistics
            # breakpoint()
            mean_ep_reward.append(self.logs[timestamp].__dict__["ep_rewards"].mean)
            mean_ep_pos_reward.append(self.logs[timestamp].__dict__["ep_pos_rewards"].mean.mean())

        # append statistics to df
        self.params_df["mean_ep_reward"] = pd.Series(mean_ep_reward, index=self.params_df.index)
        # breakpoint()
        self.params_df["mean_ep_pos_reward"] = pd.Series(mean_ep_pos_reward, index=self.params_df.index)

    def plot_mean_std(self, *args):
        for key, val in self.logs.items():
            print(key)
            val.plot_mean_std(*args)

    def plot_decorator(self, keyword="ep_rewards", window=1000, std_scale=1, save=False, zoom=2.5, loc=4):

        def stat_indexer(val, keyword):
            return val.__dict__[keyword].mean

        fig, ax, _, loc1, loc2 = print_init(False, zoom=zoom, loc=loc)

        # precompute y inset limits
        stats_max = []
        for val in self.logs.values():
            stat = stat_indexer(val, keyword)
            stats_max.append(stat.max())

        stats_max = np.array(stats_max)

        # create data structure for storing proxy values
        perf_metrics = {}

        # plot
        print("---------------------------------------------------")
        for idx, (key, val) in enumerate(self.logs.items()):
            # shorthand for the variable
            instance = self.params_df[self.params_df.timestamp == key]
            # print(f'key={key}, mean_reward={instance["mean_reward"][idx]}')

            label = instance2label(instance)

            # plot the mean of the feature
            stat = stat_indexer(val, keyword)  # calculate exp mean
            print(f'{label}, {keyword}, {stat.max()}, {stat.max() / stats_max.max()}')
            perf_metrics[label] = stat.max() #/ stats_max.max()
            x_points = self.decimate_step * np.arange(
                stat.shape[0])  # placeholder for the x points (for xtick conversion)

            ax.plot(x_points, stat, label=label, color=color4label(label))



            # if keyword == "ep_rewards":
            # plot standard deviation (uncertainty)
            # ewma_std = numpy_ewma_vectorized_v2(val.__dict__[keyword].std, window)
            # ax.fill_between(x_points, stat + std_scale * ewma_std,
            #                 stat - std_scale * ewma_std, alpha=.2, color=color4label(label))

        num_episodes=stat_indexer(val, keyword).shape[0]
        x_points = self.decimate_step * np.arange(num_episodes)
        for label, stat in perf_metrics.items():
            ax.plot(x_points,  stat*np.ones(num_episodes), "--", color=color4label(label))
        plot_postprocess(fig, ax, keyword, self.env_name, self.fig_dir, save=save)

        return perf_metrics
