import torch
import torch.nn as nn
import torch.optim as optim

from gym.spaces.utils import flatdim
from .models import A2CNet, ICMNet


class ICMAgent(nn.Module):
    def __init__(self, num_envs, num_players, action_descriptor, attn_target, attn_type, obs_space, feat_size,
                 forward_coeff, icm_beta, num_rollout, num_time, lr=1e-4):
        """
        Container class of an A2C and an ICM network, the baseline for experimenting with other curiosity-based
        methods.

        :param attn_target:
        :param attn_type:
        :param n_stack: number of frames stacked
        :param num_envs: number of parallel environments
        :param action_descriptor: size of the action space of the environment
        :param obs_space: dimensionality of the input tensor
        :param feat_size: number of the features
        :param lr: learning rate
        """
        super().__init__()

        # constants
        self.num_envs = num_envs
        self.num_players = num_players
        self.action_descriptor = action_descriptor
        self.feat_size = feat_size
        self.is_cuda = torch.cuda.is_available()


        self.features_per_object_type = [flatdim(s) for s in obs_space.spaces.values()]
        self.num_obj_types = len(obs_space.spaces.keys())

        # networks
        self.icm = ICMNet(self.num_envs, self.num_players, self.action_descriptor, attn_target, attn_type, self.features_per_object_type,
                          self.feat_size, forward_coeff, icm_beta, num_envs)
        self.a2c = A2CNet(self.num_envs, self.num_players, self.action_descriptor, self.features_per_object_type, self.feat_size, num_rollout,self.num_obj_types, num_time)

        if self.is_cuda:
            self.icm.cuda()
            self.a2c.cuda()

        # init LSTM buffers with the number of the environments
        self.a2c.set_recurrent_buffers(num_envs)

        # optimizer
        self.lr = lr
        self.optimizer = optim.Adam(list(self.icm.parameters()) + list(self.a2c.parameters()), self.lr)
