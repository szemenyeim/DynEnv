from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..utils.utils import AttentionType, AttentionTarget
from gym.spaces import MultiDiscrete, Box

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
            return torch.empty((0, 128))

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


class InOutArranger(object):

    def __init__(self, nObjectTypes, nPlayers, nTime) -> None:
        super().__init__()

        self.nTime = nTime
        self.nPlayers = nPlayers
        self.nObjectTypes = nObjectTypes
        self.indexer = Indexer(self.nObjectTypes)

    def rearrange_inputs(self, x):
        # Object counts [type x timeStep x nPlayer]
        # for each object type, for each timestep, the number of seen objects is calculated
        counts = [[[len(sightings[i]) for sightings in time] for time in x] for i in range(self.nObjectTypes)]
        objCounts = torch.tensor(
            [
                [
                    # sum of the object types for a given timestep and player
                    sum([
                        counts[obj_type][time][player] for obj_type in range(self.nObjectTypes)
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
            for i in range(self.nObjectTypes)
        ]
        inputs = [np.stack(objects) if len(objects) else np.array([]) for objects in inputs]
        return inputs, (counts, maxCount, objCounts)

    def rearrange_outputs(self, outs, countArr, device):#counts, maxCount, outs:

        counts = countArr[0]
        maxCount = countArr[1]
        objCounts = countArr[2]

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

        maxNum = outs.shape[1]
        masks = [ObsMask.createMask(counts, maxNum).to(device) for counts in objCounts]

        return outs, masks


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


# Simple embedding block for a single object type
class EmbedBlock(nn.Module):
    def __init__(self, inputs, features):
        super(EmbedBlock, self).__init__()

        self.Layer1 = nn.Linear(inputs, features // 2, bias=False)
        self.relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.LayerNorm(features // 2)
        self.Layer2 = nn.Linear(features // 2, features, bias=False)
        self.bn2 = nn.LayerNorm(features)

    def forward(self, x):

        # This happens when there is only 1 objects of this type
        if x.dim() == 1:

            # This happens when there are no sightings of this object
            if x.shape[0] == 0:
                return None

            # Unsqueeze to add batch dimension
            x = x.unsqueeze(0)

        x = self.bn1(self.relu(self.Layer1(x)))
        x = self.bn2(self.relu(self.Layer2(x)))

        return x


# Complete input layer
class InputLayer(nn.Module):
    def __init__(self, features_per_object_type, features, nEnvs, nPlayers, nObjectTypes, nTime):
        super(InputLayer, self).__init__()

        # Basic params
        self.nTime = nTime
        self.nPlayers = nPlayers * nEnvs
        self.nObjectTypes = nObjectTypes

        # Helper class for arranging tensor
        self.arranger = InOutArranger(self.nObjectTypes, self.nPlayers, self.nTime)

        # Create embedding blocks for them
        self.blocks = nn.ModuleList([EmbedBlock(input, features) for input in features_per_object_type])

    def forward(self, x):
        # Get device
        device = next(self.parameters()).device

        # Object counts [type x timeStep x nPlayer]
        # for each object type, for each timestep, the number of seen objects is calculated
        inputs, counts = self.arranger.rearrange_inputs(x)

        # Call embedding block for all object types
        outs = [block(torch.tensor(obj).to(device)) for block, obj in zip(self.blocks, inputs)]

        # Arrange objects in tensor [TimeSteps x maxObjCnt x nPlayers x featureCnt] using padding
        outs, masks = self.arranger.rearrange_outputs(outs, counts, device)

        return outs, masks


# Layer for reducing timeStep and Objects dimension via attention
class AttentionLayer(nn.Module):
    def __init__(self, feature_size, num_heads=1):
        super(AttentionLayer, self).__init__()

        self.feature_size = feature_size

        # objAtt implements self attention between objects seen in the same timestep
        self.objAtt = nn.MultiheadAttention(feature_size, num_heads)

        # Temp att attends between sightings at different timesteps
        self.tempAtt = nn.MultiheadAttention(feature_size, num_heads)

        # Relu and group norm
        self.bn = nn.LayerNorm(feature_size)

        # Confidence layer
        self.confLayer = nn.Sequential(
            nn.Linear(feature_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get device
        device = next(self.parameters()).device

        # create masks
        tensor, masks = x

        # Run self-attention
        attObj = [self.objAtt(objs, objs, objs, mask)[0] for objs, mask in zip(tensor, masks)]

        # shape = attObj[0].shape

        # Filter nans
        with torch.no_grad():
            for att in attObj:
                att[torch.isnan(att)] = 0

        # Run temporal attention
        finalAtt = attObj[0]
        finalMask = masks[0]
        for i in range(0, len(attObj) - 1):
            finalAtt = self.bn(self.tempAtt(attObj[i + 1], finalAtt, finalAtt, finalMask)[0])
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
        summed = torch.sum(finalAtt * confs, 0)
        lens = torch.sum(torch.logical_not(finalMask), 0).float().unsqueeze(1)
        lens[lens == 0] = 1.0

        return summed.div(lens)


# LSTM Layer
class LSTMLayer(nn.Module):
    def __init__(self, nPlayers, feature_size, hidden_size, nEnvs, nSteps):
        super(LSTMLayer, self).__init__()

        # Params
        self.nPlayers = nPlayers
        self.nEnvs = nEnvs
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        # Init LSTM cell
        self.cell = nn.LSTMCell(feature_size, hidden_size)

        # first call buffer generator, as reset needs the self.h buffer
        self._generate_buffers(next(self.parameters()).device, nSteps)
        self.reset()

    # Reset inner state
    def reset(self, reset_indices=None):
        device = next(self.parameters()).device

        '''with torch.no_grad():
            if reset_indices is not None and reset_indices.any():
                indices = torch.tensor(np.stack([reset_indices.cpu(),]*self.nPlayers)).permute(1,0).reshape(-1,1).squeeze()
                # Reset hidden vars
                self.h[-1][indices] = 0.0
                self.c[-1][indices] = 0.0
            else:
                self.h[-1][:] = 0.0
                self.c[-1][:] = 0.0'''

        nSteps = len(self.h)
        with torch.no_grad():
            self._generate_buffers(device, nSteps)

    def _generate_buffers(self, device, nSteps):
        self.h = [torch.zeros((self.nPlayers * self.nEnvs, self.hidden_size)).to(device) for _ in range(nSteps)]
        self.c = [torch.zeros((self.nPlayers * self.nEnvs, self.hidden_size)).to(device) for _ in range(nSteps)]
        self.x = [torch.zeros((self.nPlayers * self.nEnvs, self.feature_size)).to(device) for _ in range(nSteps)]

    # Put means and std on the correct device when .cuda() or .cpu() is called
    def _apply(self, fn):
        super(LSTMLayer, self)._apply(fn)

        self.h = [fn(h) for h in self.h]
        self.c = [fn(c) for c in self.c]

        return self

    # Forward
    def forward(self, x):
        h, c = self.cell(x, (self.h[-1], self.c[-1]))

        self.h[0].detach_()
        self.c[0].detach_()
        self.x[0].detach_()

        self.h.pop(0)
        self.c.pop(0)
        self.x.pop(0)

        self.h.append(h)
        self.c.append(c)
        self.x.append(x)

        return self.h[-1]


# Example network implementing an entire agent by simply averaging all obvervations for all timesteps
class DynEnvFeatureExtractor(nn.Module):
    def __init__(self, features_per_object_type, feature_size, num_envs, num_rollout, num_players, num_obj_types,
                 num_time):
        super().__init__()

        # feature encoding
        self.InNet = InputLayer(features_per_object_type, feature_size, num_envs, num_players, num_obj_types, num_time)
        self.AttNet = AttentionLayer(feature_size)

        self.hidden_size = feature_size
        self.LSTM = LSTMLayer(num_players, feature_size, self.hidden_size, num_envs, num_rollout)
        self.bn = nn.LayerNorm(feature_size)

    # Reset fun for lstm
    def reset(self, reset_indices=None):
        self.LSTM.reset(reset_indices)

    def forward(self, x):
        # Get embedded features
        features, objCounts = self.InNet(x)

        # Run attention
        features = self.AttNet((features, objCounts))

        # Run LSTM
        features = self.LSTM(features)
        features = self.bn(features)

        # Get actions
        return features


class AttentionNet(nn.Module):

    def __init__(self, attention_size):
        super().__init__()

        self.attention_size = attention_size

        self.attention = nn.Linear(self.attention_size, self.attention_size)

    def forward(self, target, attn=None):
        return target * F.softmax(self.attention(target if attn is None else attn), dim=-1)


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
        self.action_num_per_type = flatten([
            [num for num in action.nvec] if type(action) == MultiDiscrete else sum(action.shape)
            for action in self.action_descriptor])

        # start indx of each action type (i.e. cumsum)
        self.action_num_per_type_start_idx = np.cumsum([0, *self.action_num_per_type[:-1]])

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
        num_frames, num_action_types, num_actions = actions.shape

        # encode the current action into a one-hot vector
        # set device to that of the underlying network (it does not matter, the device of which layer is queried)
        device = current_feature.device
        action_one_hot = torch.zeros((num_frames, num_actions, np.array(self.action_num_per_type).sum())).to(device)

        # indicate with 1 the action taken by every player
        for frame_idx in range(num_frames):
            for a_type_idx in range(num_action_types):
                for a_idx, action in enumerate(actions[frame_idx, a_type_idx]):
                    action_one_hot[
                        frame_idx, a_idx, int(self.action_num_per_type_start_idx[a_type_idx] + action.item())] = 1

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
    def __init__(self, n_stack, num_players, action_descriptor, attn_target, attn_type, features_per_object_type, feat_size,
                 forward_coeff, icm_beta, num_envs):
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
        self.features_per_object_type = features_per_object_type  # pixels i.e. state
        self.feat_size = feat_size
        self.action_descriptor = action_descriptor
        self.num_actions = len(self.action_descriptor)
        self.num_envs = num_envs
        self.num_players = num_players

        self.icm_beta = icm_beta
        self.forward_coeff = forward_coeff

        # networks
        self.pred_net = AdversarialHead(self.feat_size, self.action_descriptor, attn_target,
                                        attn_type)  # goal: minimize prediction error

        self.loss_attn_flag = attn_target is AttentionTarget.ICM_LOSS and attn_type is AttentionType.SINGLE_ATTENTION
        if self.loss_attn_flag:
            self.loss_attn = AttentionNet(self.feat_size)

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

        return self._calc_loss(next_features, next_feature_pred, action_pred, action, agentFinishedMask)

    def _calc_loss(self, features, feature_preds, action_preds, actions, agentFinished):

        # If all agents finished, the loss is 0
        if not agentFinished.any():
            return torch.tensor([0, 0]).to(features.device)

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

        return (self.forward_coeff * loss_fwd, self.icm_beta * loss_inv)


class A2CNet(nn.Module):
    def __init__(self, num_envs, num_players, action_descriptor, features_per_object_type, feature_size, num_rollout, num_obj_types,
                 num_time):
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
        self.features_per_object_type = features_per_object_type
        self.feature_size = feature_size
        self.action_descriptor = action_descriptor
        self.num_players = num_players
        self.num_envs = num_envs

        self.feat_enc_net = DynEnvFeatureExtractor(self.features_per_object_type, self.feature_size, self.num_envs, num_rollout,
                                                   num_players, num_obj_types, num_time)

        self.actor = ActorLayer(self.feature_size, self.action_descriptor)
        self.critic = CriticLayer(self.feature_size)

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

        self.feat_enc_net.reset(reset_indices)

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

        return (actions, log_probs, action_probs, values,
                features)  # ide is jön egy feature bypass a self(state-ből)
