import torch
import torch.nn as nn
import torch.optim as optim

from .models import A2CNet, ICMNet


class ICMAgent(nn.Module):
    def __init__(self, num_envs, num_players, action_descriptor, attn_target, attn_type, in_size, feat_size,
                 forward_coeff, icm_beta, num_rollout, lr=1e-4):
        """
        Container class of an A2C and an ICM network, the baseline for experimenting with other curiosity-based
        methods.

        :param attn_target:
        :param attn_type:
        :param n_stack: number of frames stacked
        :param num_envs: number of parallel environments
        :param action_descriptor: size of the action space of the environment
        :param in_size: dimensionality of the input tensor
        :param feat_size: number of the features
        :param lr: learning rate
        """
        super().__init__()

        # constants
        self.num_envs = num_envs
        self.num_players = num_players
        self.action_descriptor = action_descriptor
        self.in_size = in_size
        self.feat_size = feat_size
        self.is_cuda = torch.cuda.is_available()

        # networks
        self.icm = ICMNet(self.num_envs, self.num_players, self.action_descriptor, attn_target, attn_type, self.in_size,
                          self.feat_size, forward_coeff, icm_beta, num_envs)
        self.a2c = A2CNet(self.num_envs, self.num_players, self.action_descriptor, self.in_size, self.feat_size, num_rollout)

        if self.is_cuda:
            self.icm.cuda()
            self.a2c.cuda()

        # init LSTM buffers with the number of the environments
        self.a2c.set_recurrent_buffers(num_envs)

        # optimizer
        self.lr = lr
        self.optimizer = optim.Adam(list(self.icm.parameters()) + list(self.a2c.parameters()), self.lr)
