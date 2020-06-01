import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import A2CNet
from .icm import ICMNet
from ..environment_base import RecoDescriptor


class ICMAgent(nn.Module):
    def __init__(self, hparams, num_players, action_descriptor, attn_target, attn_type, obs_space, feat_size,
                 reco_desc: RecoDescriptor, num_time, loc_feature_cnt):
        """
        Container class of an A2C and an ICM network, the baseline for experimenting with other curiosity-based
        methods.

        :param attn_target:
        :param attn_type:
        :param action_descriptor: size of the action space of the environment
        :param obs_space: dimensionality of the input tensor
        :param feat_size: number of the features
        @param hparams:
        """
        super().__init__()

        # constants
        self.num_envs = hparams.num_envs
        self.num_players = num_players
        self.action_descriptor = action_descriptor
        self.feat_size = feat_size
        self.is_cuda = torch.cuda.is_available()

        # networks
        self.icm = ICMNet(self.num_envs, self.num_players, self.action_descriptor, attn_target, attn_type,
                          self.feat_size, hparams.forward_coeff, hparams.long_horizon_coeff, hparams.icm_beta,
                          hparams.num_envs, hparams.rollout_size)
        self.a2c = A2CNet(self.num_envs, self.num_players, self.action_descriptor, obs_space,
                          self.feat_size, hparams.rollout_size, num_time, reco_desc, loc_feature_cnt)

        if self.is_cuda:
            self.icm.cuda()
            self.a2c.cuda()

        if hparams.recon_pretrained:
            self.a2c.embedder_base.load_state_dict(torch.load("models/netRec.pth"))

        # init LSTM buffers with the number of the environments
        self.a2c.set_recurrent_buffers(hparams.num_envs)

        # optimizer
        self.lr = hparams.lr
        self.optimizer = optim.Adam(list(self.icm.parameters()) + list(self.a2c.parameters()), self.lr)
