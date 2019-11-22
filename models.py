from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from curiosity.utils import AttentionType, AttentionTarget

flatten = lambda l: [item for sublist in l for item in sublist]


class ObsMask(object):
    """
    Class for calculating a mask for observations in feature space.
    `catAndPad` and `createMask` are used for the same purpose, they only use
    another representation.

    `catAndPad`: a list of tensors (in feature space) is given, which is padded
                and concatenated to have the length of the maximum number of
                observations

    `createMask`: a tensor is given, which indicates the starting index of the padding,
                  i.e. basically how many observations there are (as in `createMask`, but
                  here not the length of the list indicates the number of observations,
                  but a scalar).

    Basically, len(tensor_list[i]) = counts[i] and padded_size = max.


    """

    # Concatenate tensors and pad them to size
    @staticmethod
    def catAndPad(tensor_list: List[torch.Tensor], padded_size: int):
        """
        Given a list containing torch.Tensor instances (1st dimension shall be equal),
        they will be concatenated and the 0th dimension will be expanded

        :param tensor_list: list of tensors
        :param padded_size: target dimension of the concatenated tensors
        :return:
        """
        # If robot had no sightings
        if tensor_list is None or len(tensor_list) == 0:
            return torch.empty((0,128))

        # convert list of tensors to a single tensor
        t = torch.cat(tensor_list)

        # pad with zeros to get torch.Size([padded_size, t.shape[1]])
        return F.pad(t, pad=[0, 0, 0, padded_size - t.shape[0]], mode='constant', value=0)

    # Convert maxindices to a binary mask
    @staticmethod
    def createMask(counts: torch.Tensor, max: int) -> torch.Tensor:
        """
        Given a tensor of indices, a boolean mask is created of dimension
        torch.Size([counts.shape[0], max), where row i contains for each index j,
        where j >= counts[i]

        :param counts: tensor of indices
        :param max: max of counts
        :return: a torch.Tensor mask
        """

        mask = torch.zeros((counts.shape[0], max)).bool()
        for i, count in enumerate(counts):
            mask[i, count:] = True
        return mask


# Helper object to convert indices in a for loop quickly
class Indexer(object):
    def __init__(self, num_obj_types):
        self.prev = np.zeros(num_obj_types).astype('int64')

    def getRange(self, counts: List[List[int]], time: int, player: int, obj_type: int):
        """

        :param counts: list of lists of dimension [timestep x players]
        :param time: timestep
        :param player: player index
        :param obj_type: object type index
        :return: range between the last and new cumulative object counts
        """
        if time == 0 and player == 0:
            self.prev[obj_type] = 0
        count = counts[time][player]
        self.prev[obj_type] += count
        return range(self.prev[obj_type] - count, self.prev[obj_type])


# Outputs a certain type of action
class ActorBlock(nn.Module):
    def __init__(self, features, actions, action_type, means=None, scale=None):
        super().__init__()

        self.action_type = action_type
        # Initialize
        self.means = None
        self.scale = None

        # For continous actions a desired interval can be given [mean-range:mean+range] (otherwise [0:1])
        if means is not None:
            self.means = torch.Tensor(means)
            self.scale = torch.Tensor(scale)
            assert (len(means) == len(scale) and len(means) == actions)

        # Create layers
        self.Layer = nn.Linear(features, actions)
        self.activation = nn.Softmax(dim=1) if action_type == 'cat' else nn.Sigmoid()

    # Put means and std on the correct device when .cuda() or .cpu() is called
    def _apply(self, fn):
        super(ActorBlock, self)._apply(fn)
        if self.means is not None:
            self.means = fn(self.means)
            self.scale = fn(self.scale)
        return self

    # Forward
    def forward(self, x):

        # Forward
        x = self.activation(self.Layer(x))

        # Optional scaling
        # continuous
        if self.means is not None:
            x = (x - 0.5) * self.scale + self.means
            entropy = 0

        return x


# Outputs a certain type of action
class CriticBlock(nn.Module):
    def __init__(self, feature_size, out_size):
        super().__init__()

        # Create layers
        self.Layer = nn.Linear(feature_size, out_size)

    # Forward
    def forward(self, x):
        return self.Layer(x)


# Complete action layer for multiple action groups
class CriticLayer(nn.Module):
    def __init__(self, features):
        super().__init__()

        # Create action groups
        self.blocks = CriticBlock(features, 1)

    # Return list of actions
    def forward(self, x):
        return self.blocks(x)


# Complete action layer for multiple action groups
class ActorLayer(nn.Module):
    def __init__(self, features, actions):
        super().__init__()

        # Create action groups
        self.blocks = nn.ModuleList(
            [ActorBlock(features, action[1], action[0], action[2], action[3]) for action in actions])

    # Return list of actions
    def forward(self, x):
        outs = [block(x) for block in self.blocks]  # predict each action type
        return outs


# Simple embedding block for a single object type
class EmbedBlock(nn.Module):
    def __init__(self, inputs, features):
        super(EmbedBlock, self).__init__()

        self.Layer = nn.Linear(inputs, features)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        # This happens when there is only 1 objects of this type
        if x.dim() == 1:

            # This happens when there are no sightings of this object
            if x.shape[0] == 0:
                return None

            # Unsqueeze to add batch dimension
            x = x.unsqueeze(0)

        return self.relu(self.Layer(x))


# Complete input layer
class InputLayer(nn.Module):
    def __init__(self, inputs, features, nEnvs):
        super(InputLayer, self).__init__()

        # To be added in the future
        if len(inputs) != 4:
            raise Exception("Image and Full observation types are not yet supported")

        # Number of features of different input object types
        inputNums = inputs[-1]

        # Basic params
        self.nTime = inputs[0]
        self.nPlayers = inputs[1] * nEnvs
        self.nObjects = inputs[2]

        # Helper class for arranging tensor
        self.indexer = Indexer(self.nObjects)

        # Create embedding blocks for them
        self.blocks = nn.ModuleList([EmbedBlock(input, features) for input in inputNums])

    def forward(self, x):
        # Get device
        device = next(self.parameters()).device

        # Object counts [type x timeStep x nPlayer]
        # for each object type, for each timestep, the number of seen objects is calculated
        counts = [[[len(sightings[i]) for sightings in time] for time in x] for i in range(self.nObjects)]

        objCounts = torch.Tensor(
            [
                [
                    # sum of the object types for a given timestep and player
                    sum([
                        counts[obj_type][time][player] for obj_type in range(self.nObjects)
                    ])
                    for player in range(self.nPlayers)
                ]
                for time in range(self.nTime)
            ]).long()
        maxCount = torch.max(objCounts).item()

        # Object arrays [all objects of the same type]
        inputs = [
            flatten([
                flatten([sightings[i] for sightings in time if len(sightings[i])])
                for time in x
            ])
            for i in range(self.nObjects)
        ]
        inputs = [np.stack(objects) if len(objects) else np.array([]) for objects in inputs]

        # Call embedding block for all object types
        outs = [block(torch.Tensor(obj).to(device)) for block, obj in zip(self.blocks, inputs)]

        # Arrange objects in tensor [TimeSteps x maxObjCnt x nPlayers x featureCnt] using padding
        outs = torch.stack(
            [torch.stack(
                [
                    # in a given timestep, for a given player and object type, get the embeddings,
                    # pad them to match the length of the longest embedding tensor
                    ObsMask.catAndPad([out[self.indexer.getRange(counts[obj_type], time, player, obj_type)]
                                       for obj_type, out in enumerate(outs) if out is not None
                                       ], maxCount)
                    for player in range(self.nPlayers)
                ])
                for time in range(self.nTime)
            ])
        outs = outs.permute(0, 2, 1, 3)

        return outs, objCounts


# Layer for reducing timeStep and Objects dimension via attention
class AttentionLayer(nn.Module):
    def __init__(self, features, num_heads=1):
        super(AttentionLayer, self).__init__()

        self.features = features

        # objAtt implements self attention between objects seen in the same timestep
        self.objAtt = nn.MultiheadAttention(features, num_heads)

        # Temp att attends between sightings at different timesteps
        self.tempAtt = nn.MultiheadAttention(features, num_heads)

        # Confidence layer
        self.confLayer = nn.Sequential(
            nn.Linear(features,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get device
        device = next(self.parameters()).device

        # create masks
        tensor, objCounts = x
        maxNum = tensor.shape[1]
        masks = [ObsMask.createMask(counts, maxNum).to(device) for counts in objCounts]

        # Run self-attention
        attObj = [self.objAtt(objs, objs, objs, mask)[0] for objs, mask in zip(tensor, masks)]

        # Filter nans
        with torch.no_grad():
            for att in attObj:
                att[torch.isnan(att)] = 0

        # Run temporal attention
        finalAtt = attObj[0]
        finalMask = masks[0]
        for i in range(0, len(attObj) - 1):
            finalAtt = self.tempAtt(attObj[i + 1], finalAtt, finalAtt, finalMask)[0]
            finalMask = masks[i + 1] & finalMask
            # Filter nans
            with torch.no_grad():
                finalAtt[torch.isnan(finalAtt)] = 0

        # Mask out final attention results
        finalMask = finalMask.permute(1, 0)
        finalAtt[finalMask] = 0

        # Predict confidences for objects
        if finalAtt.shape[0] == 0:
            return torch.sum(finalAtt, 0)

        confs = self.confLayer(finalAtt)

        # Masked averaging
        summed = torch.sum(finalAtt*confs, 0)
        lens = torch.sum(torch.logical_not(finalMask), 0).float().unsqueeze(1)
        lens[lens == 0] = 1.0

        return summed.div(lens)


# LSTM Layer
class LSTMLayer(nn.Module):
    def __init__(self, nPlayers, feature, hidden, nEnvs):
        super(LSTMLayer, self).__init__()

        # Params
        self.nPlayers = nPlayers * nEnvs
        self.feature = feature
        self.hidden = hidden

        # Init LSTM cell
        self.cell = nn.LSTMCell(feature, hidden)
        self.reset()

    # Reset inner state
    def reset(self):
        # Get device
        device = next(self.parameters()).device

        # Reset hidden vars
        self.h = torch.zeros((self.nPlayers, self.hidden)).to(device)
        self.c = torch.zeros((self.nPlayers, self.hidden)).to(device)

    # Put means and std on the correct device when .cuda() or .cpu() is called
    def _apply(self, fn):
        super(LSTMLayer, self)._apply(fn)

        self.h = fn(self.h)
        self.c = fn(self.c)

        return self

    # Forward
    def forward(self, x):
        self.h, self.c = self.cell(x, (self.h, self.c))

        return self.h


# Example network implementing an entire agent by simply averaging all obvervations for all timesteps
class TestNet(nn.Module):
    def __init__(self, inputs, action, feature, nEnvs=1):
        super(TestNet, self).__init__()

        nPlayers = inputs[1] * nEnvs

        # feature encoding
        self.InNet = InputLayer(inputs, feature, nEnvs)
        self.AttNet = AttentionLayer(feature)

        self.LSTM = LSTMLayer(nPlayers, feature, feature * 2, nEnvs)

        # action prediction
        self.OutNet = ActorLayer(feature * 2, action)
        self.critic = CriticLayer(feature * 2, nPlayers)

    # Reset fun for lstm
    def reset(self):
        self.LSTM.reset()

    def forward(self, x):
        # Get embedded features
        features, objCounts = self.InNet(x)

        # Run attention
        features = self.AttNet((features, objCounts))

        # Run LSTM
        features = self.LSTM(features)

        value = self.critic(features)

        # Get actions
        return self.OutNet(features)


# Example network implementing an entire agent by simply averaging all obvervations for all timesteps
class DynEnvFeatureExtractor(nn.Module):
    def __init__(self, inputs, feature, num_envs):
        super().__init__()

        nPlayers = inputs[1]

        # feature encoding
        self.InNet = InputLayer(inputs, feature, num_envs)
        self.AttNet = AttentionLayer(feature)

        self.hidden_size = feature * 2
        self.LSTM = LSTMLayer(nPlayers, feature, self.hidden_size, num_envs)

    # Reset fun for lstm
    def reset(self):
        self.LSTM.reset()

    def forward(self, x):
        # Get embedded features
        features, objCounts = self.InNet(x)

        # Run attention
        features = self.AttNet((features, objCounts))

        # Run LSTM
        features = self.LSTM(features)

        # Get actions
        return features


class AttentionNet(nn.Module):

    def __init__(self, attention_size):
        super().__init__()

        self.attention_size = attention_size

        self.attention = nn.Linear(self.attention_size, self.attention_size)

    def forward(self, target, attn=None):
        return target * F.softmax(self.attention(target if attn is None else attn), dim=-1)


# Outputs a certain type of action
class InverseBlock(nn.Module):
    def __init__(self, features, actions, action_type, means=None, scale=None):
        super().__init__()

        self.action_type = action_type
        # Initialize
        self.means = None
        self.scale = None

        # For continous actions a desired interval can be given [mean-range:mean+range] (otherwise [0:1])
        if means is not None:
            self.means = torch.Tensor(means)
            self.scale = torch.Tensor(scale)
            assert (len(means) == len(scale) and len(means) == actions)

        # Create layers
        self.fc_hidden = 256
        self.fc1 = nn.Linear(features * 2, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, actions)
        self.activation = nn.Softmax(dim=1) if action_type == 'cat' else nn.Sigmoid()

    # Put means and std on the correct device when .cuda() or .cpu() is called
    def _apply(self, fn):
        super()._apply(fn)
        if self.means is not None:
            self.means = fn(self.means)
            self.scale = fn(self.scale)
        return self

    # Forward
    def forward(self, x):

        # Forward
        x = self.activation(self.fc1(x))

        # Optional scaling
        # continuous
        if self.means is not None:
            x = (x - 0.5) * self.scale + self.means
            entropy = 0

        return x


# Complete action layer for multiple action groups
class InverseNet(nn.Module):
    def __init__(self, features, actions):
        super().__init__()

        # Create action groups
        self.blocks = nn.ModuleList(
            [InverseBlock(features, action[1], action[0], action[2], action[3]) for action in actions])

    # Return list of actions
    def forward(self, x):
        outs = [block(x) for block in self.blocks]  # predict each action type
        return outs


# class InverseNet(nn.Module):
#     def __init__(self, num_actions, feat_size=288):
#         """
#         Network for the inverse dynamics
#
#         :param num_actions: number of actions, pass env.action_space.n
#         :param feat_size: dimensionality of the feature space (scalar)
#         """
#         super().__init__()
#
#         # constants
#         self.feat_size = feat_size
#         self.fc_hidden = 256
#         self.num_actions = num_actions
#
#         # layers
#         self.fc1 = nn.Linear(self.feat_size * 2, self.fc_hidden)
#         self.fc2 = nn.Linear(self.fc_hidden, self.num_actions)
#
#     def forward(self, x):
#         """
#         In: torch.cat((phi(s_t), phi(s_{t+1}), 1)
#             Current and next states transformed into the feature space,
#             denoted by phi().
#
#         Out: \hat{a}_t
#             Predicted action
#
#         :param x: input data containing the concatenated current and next states, pass
#                   torch.cat((phi(s_t), phi(s_{t+1}), 1)
#         :return:
#         """
#         return self.fc2(self.fc1(x))
#

class ForwardNet(nn.Module):

    def __init__(self, in_size):
        """
        Network for the forward dynamics

        :param in_size: size(feature_space) + size(action_space)
        """
        super().__init__()

        # constants
        self.in_size = in_size
        self.fc_hidden = 140
        self.out_size = 256

        # layers

        self.fc1 = nn.Linear(self.in_size, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, self.out_size)

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
        return self.fc2(self.fc1(x))


class AdversarialHead(nn.Module):
    def __init__(self, feat_size, action_descriptor, attn_target, attention_type):
        """
        Network for exploiting the forward and inverse dynamics

        :param attn_target:
        :param attention_type:
        :param feat_size: size of the feature space
        :param action_descriptor: size of the action space, pass env.action_space.n
        """
        super().__init__()

        # constants
        self.feat_size = feat_size
        self.action_descriptor = action_descriptor
        # number of discrete actions per type
        self.action_num_per_type = [action[1] for action in self.action_descriptor]

        # start indx of each action type (i.e. cumsum)
        self.action_num_per_type_start_idx = np.cumsum([0, *self.action_num_per_type[:-1]])

        # networks
        self.fwd_net = ForwardNet(2*self.feat_size + np.array(self.action_num_per_type).sum())
        self.inv_net = ActorLayer(self.feat_size * 4, self.action_descriptor)

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
        num_frames, num_action_types, num_actions = actions.shape

        # one-hot placeholder vector
        device = current_feature.device
        action_one_hot = torch.zeros((num_frames, num_actions, np.array(self.action_num_per_type).sum())).to(device)

        # indicate with 1 the action taken by every player
        for frame_idx in range(num_frames):
            for a_type_idx in range(num_action_types):
                for a_idx, action in enumerate(actions[frame_idx, a_type_idx]):
                    action_one_hot[frame_idx, a_idx, int(self.action_num_per_type_start_idx[a_type_idx] + action.item())] = 1


        # encode the current action into a one-hot vector
        # set device to that of the underlying network (it does not matter, the device of which layer is queried)

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


class ICMNet(nn.Module):
    def __init__(self, n_stack, num_players, action_descriptor, attn_target, attn_type, in_size, feat_size, num_envs=1):
        """
        Network implementing the Intrinsic Curiosity Module (ICM) of https://arxiv.org/abs/1705.05363

        :param num_players:
        :param num_envs:
        :param n_stack: number of frames stacked
        :param action_descriptor: dimensionality of the action space, pass env.action_space.n
        :param attn_target:
        :param attn_type:
        :param in_size: input size of the AdversarialHeads
        :param feat_size: size of the feature space
        """
        super().__init__()

        # constants
        self.in_size = in_size  # pixels i.e. state
        self.feat_size = feat_size
        self.action_descriptor = action_descriptor
        self.num_actions = len(self.action_descriptor)
        self.num_envs = num_envs
        self.num_players = num_players

        self.prev_features = None

        # networks
        self.pred_net = AdversarialHead(self.feat_size, self.action_descriptor, attn_target,
                                        attn_type)  # goal: minimize prediction error

        self.loss_attn_flag = attn_target is AttentionTarget.ICM_LOSS and attn_type is AttentionType.SINGLE_ATTENTION
        if self.loss_attn_flag:
            self.loss_attn = AttentionNet(self.feat_size)

    def forward(self, features, action):
        """

        feature: current encoded state
        next_feature: next encoded state

        :param features: tensor of the states
        :param action: current action
        :return:
        """

        """Predict fwd & inv dynamics"""
        current_features = features[:-1,:,:]
        next_features =features[1:,:,:]
        next_feature_pred, action_pred = self.pred_net(current_features, next_features, action)


        return self._calc_loss(next_features, next_feature_pred, action_pred, action)

    def _calc_loss(self, features, feature_preds, action_preds, actions):

        # forward loss
        # measure of how good features can be predicted
        if not self.loss_attn_flag:
            loss_fwd = F.mse_loss(feature_preds, features)
        # else:
        #     loss_fwd = self.loss_attn(F.mse_loss(feature_preds, features, reduction="none"), features).mean()
        # inverse loss
        # how good is the action estimate between states
        actions = actions.permute(1,2,0)
        loss_inv = torch.stack([F.cross_entropy(a_pred.permute(1,2,0), a.long()) for (a_pred, a) in zip(action_preds, actions)]).mean()

        return loss_fwd + loss_inv


class A2CNet(nn.Module):
    def __init__(self, num_envs, num_players, action_descriptor, in_size, feature_size):
        """
        Implementation of the Advantage Actor-Critic (A2C) network

        :param num_envs: 
        :param n_stack: number of frames stacked
        :param action_descriptor: size of the action space, pass env.action_space.n
        :param in_size: input size of the LSTMCell of the FeatureEncoderNet
        """
        super().__init__()

        print("in case of multiple envs, reset_recurrent_buffers shall be revisited to handle non-simultaneous resets")
        # constants
        self.in_size = in_size  # in_size
        self.feature_size = feature_size
        self.action_descriptor = action_descriptor
        self.num_players = num_players*2
        self.num_envs = num_envs

        self.feat_enc_net = DynEnvFeatureExtractor(self.in_size, self.feature_size, self.num_envs)

        self.actor = ActorLayer(self.feature_size * 2, self.action_descriptor)
        self.critic = CriticLayer(self.feature_size * 2)

    def set_recurrent_buffers(self, buf_size):
        """
        Initializes LSTM buffers with the proper size,
        should be called after instatiation of the network.

        :param buf_size: size of the recurrent buffer
        :return:
        """
        self.feat_enc_net.reset()

    def reset_recurrent_buffers(self, reset_indices=None):
        """

        :param reset_indices: boolean numpy array containing True at the indices which
                              should be reset
        :return:
        """

        self.feat_enc_net.reset()

    def forward(self, state):
        """

        feature: current encoded state

        :param state: current state
        :return:
        """

        # encode the state
        feature = self.feat_enc_net(state)

        # calculate policy and value function
        policy = self.actor(feature)
        value = self.critic(feature)

        return policy, value, feature

    def get_action(self, state):
        """
        Method for selecting the next action

        :param state: current state
        :return: tuple of (action, log_prob_a_t, value)
        """

        """Evaluate the A2C"""
        policies, values, features = self(state)  # use A3C to get policy and value

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical  distribution represented by these probabilities
        action_probs = [F.softmax(player_policy, dim=-1) for player_policy in policies]
        cats = [Categorical(a_prob) for a_prob in action_probs]
        actions = [cat.sample() for cat in cats]
        log_probs = [cat.log_prob(a) for (cat, a) in zip(cats, actions)]
        entropies = [cat.entropy().mean() for cat in cats]

        return (actions, log_probs, entropies, values,
                features)  # ide is jön egy feature bypass a self(state-ből)
