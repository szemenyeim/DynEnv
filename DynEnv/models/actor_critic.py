import torch
from gym.spaces import Box
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .models import DynEvnEncoder
from ..utils.utils import flatten


class A2CNet(nn.Module):
    def __init__(self, num_envs, num_players, action_descriptor, obs_space, feature_size, num_rollout, num_time,
                 reco_desc, loc_feature_cnt):
        """
        Implementation of the Advantage Actor-Critic (A2C) network

        :param num_envs:
        :param n_stack: number of frames stacked
        :param action_descriptor: size of the action space, pass env.action_space.n
        :param features_per_object_type: input size of the LSTMCell of the FeatureEncoderNet
        """
        super().__init__()

        print("in case of multiple envs, reset_recurrent_buffers shall be revisited to handle non-simultaneous resets")
        # constants
        self.feature_size = feature_size
        self.action_descriptor = action_descriptor
        self.num_players = num_players
        self.num_envs = num_envs
        actionCnt = sum([sum(space.shape) for space in action_descriptor.spaces])

        self.embedder_base = DynEvnEncoder(feature_size, num_envs, num_rollout, num_players, num_time,
                                           obs_space.spaces[1], obs_space.spaces[0], reco_desc, actionCnt, loc_feature_cnt)

        self.actor = ActorLayer(self.feature_size * 2, self.action_descriptor)
        self.critic = CriticLayer(self.feature_size * 2)

    def set_recurrent_buffers(self, buf_size):
        """
        Initializes LSTM buffers with the proper size,
        should be called after instatiation of the network.

        :param buf_size: size of the recurrent buffer
        :return:
        """
        self.embedder_base.reset()

    def reset_recurrent_buffers(self, reset_indices=None):
        """

        :param reset_indices: boolean numpy array containing True at the indices which
                              should be reset
        :return:
        """

        self.embedder_base.reset(reset_indices=reset_indices)

    def forward(self, state):
        """

        feature: current encoded state

        :param state: current state
        :return:
        """

        state = (state[0][..., 0], state[0][..., 1], state[1])

        # encode the state
        features = self.embedder_base(state)

        feature = torch.cat(features[0:2], dim=1)

        # calculate policy and value function
        policy = self.actor(feature)
        value = self.critic(feature)

        return policy, value, feature, features[2]

    def get_recon_loss(self, feature, targets):
        return self.embedder_base.reconstructor(feature, targets)

    def get_action(self, state):
        """
        Method for selecting the next action

        :param state: current state
        :return: tuple of (action, log_prob_a_t, value)
        """

        """Evaluate the A2C"""
        policies, values, features, pos = self(state)  # use A3C to get policy and value

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical  distribution represented by these probabilities
        action_probs = [F.softmax(player_policy, dim=-1) for player_policy in policies]
        cats = [Categorical(a_prob) for a_prob in action_probs]
        actions = [cat.sample() for cat in cats]
        log_probs = [cat.log_prob(a) for (cat, a) in zip(cats, actions)]

        return (actions, log_probs, action_probs, values,
                features, pos)  # ide is jön egy feature bypass a self(state-ből)


# Complete action layer for multiple action groups
class ActorLayer(nn.Module):
    def __init__(self, features, actions):
        super().__init__()

        # Create action groups
        self.blocks = nn.ModuleList(
            [ActorBlock(features, action) for action in actions])

    # Return list of actions
    def forward(self, x):
        outs = flatten([block(x) for block in self.blocks])  # predict each action type
        return outs


# Outputs a certain type of action
class ActorBlock(nn.Module):
    def __init__(self, features, action_space):
        super().__init__()

        self.cont = type(action_space) == Box

        # For continous actions a desired interval can be given [mean-range:mean+range] (otherwise [0:1])
        if self.cont:
            self.shape = sum(action_space.shape)
            mean = (action_space.high + action_space.low) * 0.5
            scale = (action_space.high - action_space.low) * 0.5
            self.means = torch.tensor(mean)
            self.scale = torch.tensor(scale)
            self.Layer = nn.Linear(features, self.shape)
            self.activation = nn.Sigmoid()
        else:
            actionNum = action_space.nvec
            self.Layer = nn.ModuleList([nn.Linear(features, num) for num in actionNum])

    # Put means and std on the correct device when .cuda() or .cpu() is called
    def _apply(self, fn):
        super(ActorBlock, self)._apply(fn)
        if self.cont:
            self.means = fn(self.means)
            self.scale = fn(self.scale)
        return self

    # Forward
    def forward(self, x):

        if self.cont:
            x = self.Layer(x)
            x = self.activation(x)
            x = (x - 0.5) * self.scale + self.means
        else:
            x = [l(x) for l in self.Layer]

        return x


# Complete action layer for multiple action groups
class CriticLayer(nn.Module):
    def __init__(self, features):
        super().__init__()

        # Create action groups
        self.blocks = CriticBlock(features, 1)

    # Return list of actions
    def forward(self, x):
        return self.blocks(x)


# Outputs a certain type of action
class CriticBlock(nn.Module):
    def __init__(self, feature_size, out_size):
        super().__init__()

        # Create layers
        self.Layer1 = nn.Linear(feature_size, feature_size // 2)
        self.Layer2 = nn.Linear(feature_size // 2, out_size)
        self.relu = nn.LeakyReLU(0.1)
        self.bn = nn.LayerNorm(feature_size // 2)

    # Forward
    def forward(self, x):
        return self.Layer2(self.bn(self.relu(self.Layer1(x))))
