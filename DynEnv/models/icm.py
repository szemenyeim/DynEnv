import numpy as np
import torch
from gym.spaces import MultiDiscrete
from torch import nn as nn
from torch.nn import functional as F

from ..models.actor_critic import ActorLayer
from ..models.loss_descriptors import ICMLosses
from ..utils import AttentionTarget, AttentionType
from ..utils.utils import flatten


class ICMNet(nn.Module):
    def __init__(self, n_stack, num_players, action_descriptor, attn_target, attn_type, feat_size, forward_coeff,
                 long_horizon_coeff, icm_beta, num_envs, num_rollout):
        """
        Network implementing the Intrinsic Curiosity Module (ICM) of https://arxiv.org/abs/1705.05363

        :param num_players:
        :param num_envs:
        :param n_stack: number of frames stacked
        :param action_descriptor: dimensionality of the action space, pass env.action_space.n
        :param attn_target:
        :param attn_type:
        :param features_per_object_type: input size of the AdversarialHeads
        :param feat_size: size of the feature space
        """
        super().__init__()

        # constants
        self.feat_size = feat_size * 2
        self.action_descriptor = action_descriptor
        self.num_actions = len(self.action_descriptor)
        self.num_envs = num_envs
        self.num_players = num_players
        self.num_rollout = num_rollout

        self.icm_beta = icm_beta
        self.forward_coeff = forward_coeff
        self.long_horizon_coeff = long_horizon_coeff

        # networks
        self.pred_net = ICMDynamics(self.feat_size, self.action_descriptor, attn_target,
                                    attn_type)  # goal: minimize prediction error

        self.loss_attn_flag = attn_target is AttentionTarget.ICM_LOSS and attn_type is AttentionType.SINGLE_ATTENTION
        if self.loss_attn_flag:
            self.loss_attn = AttentionNet(self.feat_size)

        self.loss_long_horizon_curiosity = LongHorizonCuriosityLoss(self.feat_size)
        self.long_horizon_fwd_net = LongHorizonForwardNet(self.feat_size, self.action_descriptor, self.num_rollout)

    def forward(self, features, action, agentFinished):
        """

        feature: current encoded state
        next_feature: next encoded state

        :param features: tensor of the states
        :param action: current action
        :return:
        """

        """Predict fwd & inv dynamics"""
        current_features = features[:-1, :, :]
        next_features = features[1:, :, :]
        next_feature_pred, action_pred = self.pred_net(current_features, next_features, action)

        # Agent finished status to mask inverse and forward losses
        agentFinishedMask = torch.logical_not(agentFinished)

        long_horizon_feature_preds = self.long_horizon_fwd_net(features[0,...], action)

        return self._calc_loss(next_features, next_feature_pred, long_horizon_feature_preds, action_pred, action,
                               agentFinishedMask)

    def _calc_loss(self, features, feature_preds, long_horizon_feature_preds, action_preds, actions, agentFinished):

        # If all agents finished, the loss is 0
        if not agentFinished.any():
            loss = torch.zeros(1).to(features.device)
            icm_losses = ICMLosses(forward=self.forward_coeff * loss, inverse=self.icm_beta * loss,
                                   long_horizon_forward=self.long_horizon_coeff * loss)
            icm_losses.prepare_losses()

            return icm_losses

        # forward loss
        # measure of how good features can be predicted
        if not self.loss_attn_flag:
            loss_fwd = F.mse_loss(feature_preds[agentFinished], features[agentFinished])
        else:
            loss_fwd = self.loss_attn(F.mse_loss(feature_preds, features, reduction="none"), features).mean()
        # inverse loss
        # how good is the action estimate between states
        actions = actions.permute(1, 0, 2)
        # print(torch.min(action_preds[0]),torch.max(action_preds[0]))
        losses = [F.cross_entropy(a_pred.permute(0, 2, 1), a.long(), reduction='none')[agentFinished].mean() for
                  (a_pred, a) in zip(action_preds, actions)]
        loss_inv = torch.stack(losses).mean()

        # long-horizon curiosity loss
        loss_long_fwd = self.loss_long_horizon_curiosity(long_horizon_feature_preds, features)

        icm_losses = ICMLosses(forward=self.forward_coeff * loss_fwd, inverse=self.icm_beta * loss_inv,
                               long_horizon_forward=self.long_horizon_coeff*loss_long_fwd)
        icm_losses.prepare_losses()

        return icm_losses


class ForwardNet(nn.Module):

    def __init__(self, feat_size, action_size):
        """
        Network for the forward dynamics

        :param in_size: size(feature_space) + size(action_space)
        """
        super().__init__()

        # constants
        self.in_size = feat_size + action_size
        self.fc_hidden = 140
        self.out_size = feat_size

        # layers

        self.fc1 = nn.Linear(self.in_size, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, self.out_size)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        """
        In: torch.cat((phi(s_t), a_t), 1)
            Current state transformed into the feature space,
            denoted by phi() and current action

        Out: \hat{phi(s_{t+1})}
            Predicted next state (in feature space)

        :param x: input data containing the concatenated current state in feature space
                  and the current action, pass torch.cat((phi(s_t), a_t), 1)
        :return:
        """
        return self.fc2(self.relu(self.fc1(x)))


class DynamicsBase(nn.Module):
    def __init__(self, feat_size, action_descriptor):
        super().__init__()

        self.action_descriptor = action_descriptor
        self.feat_size = feat_size

        self._calc_action_nums_per_type()

    def _calc_action_nums_per_type(self):
        # number of discrete actions per type
        self.action_num_per_type = flatten([
            [num for num in action.nvec] if type(action) == MultiDiscrete else sum(action.shape)
            for action in self.action_descriptor])

        # start idx of each action type (i.e. cumsum)
        self.action_num_per_type_start_idx = np.cumsum([0, *self.action_num_per_type[:-1]])

    def _actions2onehot(self, actions):
        num_frames, num_action_types, num_actions = actions.shape
        # encode the current action into a one-hot vector
        # set device to that of the underlying network (it does not matter, the device of which layer is queried)
        device = actions.device
        action_one_hot = torch.zeros((num_frames, num_actions, np.array(self.action_num_per_type).sum())).to(device)
        # indicate with 1 the action taken by every player
        for frame_idx in range(num_frames):
            for a_type_idx in range(num_action_types):
                for a_idx, action in enumerate(actions[frame_idx, a_type_idx]):
                    action_one_hot[
                        frame_idx, a_idx, int(self.action_num_per_type_start_idx[a_type_idx] + action.item())] = 1
        return action_one_hot


class ICMDynamics(DynamicsBase):
    def __init__(self, feat_size, action_descriptor, attn_target, attention_type):
        """
        Network for exploiting the forward and inverse dynamics

        :param attn_target:
        :param attention_type:
        :param feat_size: size of the feature space
        :param action_descriptor: size of the action space, pass env.action_space.n
        """
        super().__init__(feat_size, action_descriptor)

        # networks
        self.fwd_net = ForwardNet(self.feat_size, np.array(self.action_num_per_type).sum())
        self.inv_net = ActorLayer(self.feat_size * 2, self.action_descriptor)

        # attention
        self.attention_type = attention_type
        self.attn_target = attn_target

        if self.attn_target is AttentionTarget.ICM:
            if self.attention_type == AttentionType.SINGLE_ATTENTION:
                self.fwd_att = AttentionNet(self.feat_size + len(self.action_descriptor))
                self.inv_att = AttentionNet(2 * self.feat_size)

    def forward(self, current_feature, next_feature, actions):
        """

        :param current_feature: current encoded state
        :param next_feature: next encoded state
        :param actions: current action
        :return: next_feature_pred (estimate of the next state in feature space),
                 action_pred (estimate of the current action)
        """

        """Forward dynamics"""
        action_one_hot = self._actions2onehot(actions)

        if self.attn_target is AttentionTarget.ICM:
            if self.attention_type == AttentionType.SINGLE_ATTENTION:
                fwd_in = self.fwd_att(torch.cat((current_feature, action_one_hot), 2))
        else:
            fwd_in = torch.cat((current_feature, action_one_hot), 2)

        next_feature_pred = self.fwd_net(fwd_in)

        """Inverse dynamics"""
        # predict the action between s_t and s_t1
        if self.attn_target is AttentionTarget.ICM:
            if self.attention_type == AttentionType.SINGLE_ATTENTION:
                inv_in = self.inv_att(torch.cat((current_feature, next_feature), 2))
            elif self.attention_type == AttentionType.DOUBLE_ATTENTION:
                inv_in = torch.cat((self.inv_cur_feat_att(current_feature), self.inv_next_feat_att(next_feature)), 1)
        else:
            inv_in = torch.cat((current_feature, next_feature), 2)

        action_pred = self.inv_net(inv_in)

        return next_feature_pred, action_pred


class AttentionNet(nn.Module):

    def __init__(self, attention_size):
        super().__init__()

        self.attention_size = attention_size

        self.attention = nn.Linear(self.attention_size, self.attention_size)

    def forward(self, target, attn=None):
        return target * (F.softmax(self.attention(target if attn is None else attn), dim=-1))


class LongHorizonCuriosityLoss(nn.Module):
    def __init__(self, attention_size) -> None:
        """
        Class for calculating the attention-corrected loss for
        long-horizon forward prediction in curiosity.

        The base idea is the following (for implementation details, check the forward method):
            - it would be good to have a multi-step lookahead in the curiosity module
            - using previous (predicted) states for further prediction should be good, as it is
              expected that temporally near states are similar to each other
            - the problem is that using a predicted state as a base for further prediction
             would mean that errors would be accumulated
            - with trainable weights (attention layer) we intend compensate for this error accumulation
        """

        super().__init__()

        # todo: use only 1 attention layer or so much as the prediction length (rollout)?
        self.attention = AttentionNet(attention_size)

    def forward(self, pred_states, true_states):
        # device
        device = pred_states.device

        # default values (set to the correct device)
        mse_loss = torch.tensor(0.0).to(device)
        weight = torch.tensor(1.0).to(device)

        # loop over timesteps in the rollout
        for (pred, true) in zip(pred_states, true_states):
            # calculate MSE without reduction (needed for the attention layer)
            mse_step_loss = F.mse_loss(pred, true, reduction="none")
            # accumulate
            mse_loss += (weight * mse_step_loss).mean()

            # calculate "lookahead weight"
            # this is the "confidence" compensating for the "drift" in long term forward prediction
            # i.e. as the prediction for s_t will not be perfect, if that prediction is used to predict for t+1, t+2, ...
            # then a prediction of the same quality would result in higher losses (as the starting point - which is the
            # prediction of the previous step - is not perfect), which would be good to eliminate
            weight = self.attention(mse_step_loss)

        return mse_loss


class LongHorizonForwardNet(DynamicsBase):

    def __init__(self, feat_size, action_descriptor, num_pred_steps):
        """
        Network for the long-horizon forward dynamics

        :param in_size: size(feature_space) + size(action_space)
        """
        super().__init__(feat_size, action_descriptor)

        # constants
        self.feat_size = feat_size
        self.action_size = np.array(self.action_num_per_type).sum()

        self.num_pred_steps = num_pred_steps

        # layers
        self.fwd_nets = nn.ModuleList([ForwardNet(self.feat_size, self.action_size) for _ in range(self.num_pred_steps)])


    def forward(self, current_feature, actions):
        """
        @param current_feature: current state transformed into the feature space,
        @param actions: (predicted) actions for the rollout #todo: use predicted or real actions? -> I would opt for the former

        @return: predicted next states (in feature space) for the whole rollout
        """

        one_hot_actions = self._actions2onehot(actions)

        pred_features = [current_feature]
        for fwd_net, one_hot_action in zip(self.fwd_nets, one_hot_actions):
            pred_features.append(fwd_net(torch.cat((pred_features[-1], one_hot_action), 1)))

        return torch.stack(pred_features[1:])
