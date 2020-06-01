import itertools
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, MultiBinary, Discrete
from gym.spaces import flatdim
from torch import nn as nn

from .loss_descriptors import LocalizationLosses, ReconLosses
from ..environment_base import RecoDescriptor
from ..utils.utils import build_targets, flatten


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

        # Transformer
        self.transformer = nn.Sequential(
            nn.Linear(6, self.feature_size),
            nn.Tanh()
        )

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

        nSteps = 1
        with torch.no_grad():
            self._generate_buffers(device, nSteps)

    def detach(self):
        self.h = self.h.detach()
        self.c = self.c.detach()

    def get_states(self):
        return [self.h.clone(), self.c.clone()]

    def set_states(self, states):
        self.h = states[0].clone()
        self.c = states[1].clone()

    def set_state(self, state):
        device = next(self.parameters()).device
        x = state.to(device)
        self.c = self.transformer(x)
        self.f = self.c
        '''with torch.no_grad():
            self.c = state.to(device)
            self.h = state.to(device)'''

    def _generate_buffers(self, device, nSteps):
        self.h = torch.zeros((self.nPlayers * self.nEnvs, self.hidden_size)).to(device)
        self.c = torch.zeros((self.nPlayers * self.nEnvs, self.hidden_size)).to(device)
        self.x = torch.zeros((self.nPlayers * self.nEnvs, self.feature_size)).to(device)

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
            return torch.empty((0, padded_size))

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
        # reorder observations
        numT = len(x[0])
        x = [list(itertools.chain.from_iterable([x[env][time] for env in range(len(x))])) for time in
             range(numT)]

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

    def rearrange_outputs(self, outs, countArr, device):  # counts, maxCount, outs:

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
class RecurrentTemporalAttention(nn.Module):
    def __init__(self, feature_size, num_heads=1):
        super(RecurrentTemporalAttention, self).__init__()

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
        for att in attObj:
            att[torch.isnan(att)] = 0

        # Run temporal attention
        finalAtt = attObj[0]
        finalMask = masks[0]
        for i in range(0, len(attObj) - 1):
            finalAtt = self.bn(self.tempAtt(attObj[i + 1], finalAtt, finalAtt, finalMask)[0])
            finalMask = masks[i + 1] & finalMask
            # Filter nans
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


class ReconNet(nn.Module):

    def __init__(self, inplanes, reco_desc: RecoDescriptor):
        super().__init__()

        self.reco_desc = reco_desc
        self.inplanes = inplanes
        self.ignore_threses = [0.01, 0.04, 0.16]  # [0.0025, 0.01, 0.04] #

        self.numChannels = 0
        self._create_class_defs()

        self.nn = nn.Sequential(
            nn.ConvTranspose2d(inplanes, inplanes * 2, self.reco_desc.featureGridSize),
            nn.LeakyReLU(0.1),
            nn.LayerNorm([inplanes * 2, self.reco_desc.featureGridSize[0], self.reco_desc.featureGridSize[1]]),
            # nn.BatchNorm2d(inplanes * 2),
            nn.Conv2d(inplanes * 2, self.numChannels, 1)
        )

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def _create_class_defs(self):

        self.PosIndices = []
        self.MSEIndices = []
        self.BCEIndices = []
        self.CEIndices = []

        self.classDefs = []

        for classDef in self.reco_desc.fullStateSpace:
            currClassDef = []
            for i in range(classDef.numItemsPerGridCell):
                for j, data in enumerate(classDef.space.spaces.items()):
                    key, x = data
                    t = type(x)
                    if t == Discrete:
                        self.CEIndices += range(self.numChannels, self.numChannels + x.n)
                        self.numChannels += x.n
                        currClassDef += ['cat', ]
                    elif t == MultiBinary:
                        self.BCEIndices += range(self.numChannels, self.numChannels + x.n)
                        self.numChannels += x.n
                        currClassDef += ['bin', ] * x.n if key != "confidence" else ['conf', ]
                    elif t == Box:
                        if key == 'position':
                            self.PosIndices += range(self.numChannels, self.numChannels + x.shape[0])
                        else:
                            self.MSEIndices += range(self.numChannels, self.numChannels + x.shape[0])
                        self.numChannels += x.shape[0]
                        currClassDef += [key, ] * x.shape[0]
            self.classDefs.append([classDef.numItemsPerGridCell, currClassDef])

    def forward(self, x, targets, seens):

        x = x.reshape([-1, self.inplanes, 1, 1])

        if targets is not None:
            targets = flatten(list(targets))

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # Initialize losses
        reco_losses = ReconLosses(num_classes=len(self.classDefs), num_thresh=len(self.ignore_threses))
        if x.is_cuda:
            reco_losses.cuda()

        preds = self.nn(x)

        # preds[:, self.MSEIndices] = torch.tanh(preds[:, self.MSEIndices])
        preds[:, self.BCEIndices] = torch.sigmoid(preds[:, self.BCEIndices])
        # preds[:, self.PosIndices] = torch.sigmoid(preds[:, self.PosIndices])
        '''with torch.no_grad():
            preds[:, self.PosIndices] = torch.clamp(preds[:, self.PosIndices], min=-3, max=3)
        preds[:,self.PosIndices] = (preds[:,self.PosIndices] + 3) / 6'''

        predOffs = 0
        nGy, nGx = self.reco_desc.featureGridSize

        for classInd, (cDef, predInfo) in enumerate(zip(self.classDefs, self.reco_desc.targetDefs)):
            nA = cDef[0]
            elemIDs = cDef[1]
            nElems = len(elemIDs)
            lenA = nElems // nA
            elemDesc = elemIDs[:lenA]

            numTar = targets[0][classInd].shape[0]

            seen = seens[classInd].view(-1, numTar, 1, 1)
            if x.is_cuda:
                seen = seen.cuda()

            cPreds = preds[:, predOffs:predOffs + nElems]
            cPreds = cPreds.view((-1, nA, lenA, nGy, nGx)).permute((0, 1, 3, 4, 2)).contiguous()
            predOffs += nElems

            ind = [i for i, x in enumerate(elemDesc) if x == 'position']
            x = cPreds[..., ind[0]]
            y = cPreds[..., ind[1]]
            ind = [i for i, x in enumerate(elemDesc) if x == 'conf']
            pred_conf = cPreds[..., ind[0]]
            ind = [i for i, x in enumerate(elemDesc) if x == 'cat']
            pred_class = cPreds[..., ind] if ind else None
            ind = [i for i, x in enumerate(elemDesc) if x == 'bin']
            pred_bins = cPreds[..., ind] if ind else None
            ind = [i for i, x in enumerate(elemDesc) if x not in ['position', 'bin', 'conf', 'cat']]
            pred_cont = cPreds[..., ind] if ind else None

            # Calculate offsets for each grid
            grid_x = torch.arange(nGx).repeat(nGy, 1).view([1, 1, nGy, nGx]).type(FloatTensor)
            grid_y = torch.arange(nGy).repeat(nGx, 1).t().view([1, 1, nGy, nGx]).type(FloatTensor)

            # Add offset and scale with anchors
            pred_coords = torch.stack([x.detach() + grid_x, y.detach() + grid_y], -1)

            if targets is not None:
                nGT, nCorrect, mask, conf_mask, tx, ty, tcont, tbin, tconf, tcls, corr = build_targets(
                    pred_coords=pred_coords.cpu().detach(),
                    pred_conf=pred_conf.cpu().detach(),
                    targets=targets,
                    num_anchors=nA,
                    grid_size_y=nGy,
                    grid_size_x=nGx,
                    ignore_threses=self.ignore_threses,
                    predInfo=predInfo,
                    classInd=classInd,
                    seen=seen
                )

                nProposals = int((pred_conf > 0.5).sum().item())
                nCorrPrec = [int((c).sum().item()) for c in corr]
                nObjs = seen.numel()
                reco_losses.update_stats(nCorrect=nCorrect, nCorrectPrec=nCorrPrec, nPred=nProposals, nTotal=nGT,
                                         nObjs=nObjs,
                                         idx=classInd)

                # Handle masks
                mask = mask.type(ByteTensor).bool()
                conf_mask = conf_mask.type(ByteTensor).bool()

                # Handle target variables
                tx = tx.type(FloatTensor)
                ty = ty.type(FloatTensor)
                tbin = tbin.type(FloatTensor)
                tcont = tcont.type(FloatTensor)
                tconf = tconf.type(FloatTensor)
                tcls = tcls.type(LongTensor)

                # Get conf mask where gt and where there is no gt
                conf_mask_true = mask
                conf_mask_false = conf_mask ^ mask

                # Mask outputs to ignore non-existing objects
                if mask.any():
                    loss_x = self.mse_loss(x[mask], tx[mask])
                    loss_y = self.mse_loss(y[mask], ty[mask])
                    loss_cont = self.mse_loss(pred_cont[mask], tcont[mask]) if pred_cont is not None else torch.tensor(
                        0).type(FloatTensor)
                    loss_bin = self.bce_loss(pred_bins[mask], tbin[mask]) if pred_bins is not None else torch.tensor(
                        0).type(FloatTensor)
                    loss_conf1 = self.bce_loss(pred_conf[conf_mask_false],
                                               tconf[conf_mask_false]) if conf_mask_false.any() else torch.tensor(
                        0).type(FloatTensor)
                    loss_conf2 = self.bce_loss(pred_conf[conf_mask_true],
                                               tconf[conf_mask_true]) if conf_mask_true.any() else torch.tensor(0).type(
                        FloatTensor)
                    loss_conf = 1 * loss_conf1 + loss_conf2
                    loss_cls = self.ce_loss(pred_class[mask], tcls[mask]) if pred_class is not None else torch.tensor(
                        0).type(FloatTensor)
                else:
                    loss_x = loss_y = loss_cont = loss_bin = loss_cls = loss_conf = torch.tensor(0.0).cuda()

                reco_losses.update_losses(loss_x, loss_y, loss_conf, loss_cont, loss_bin, loss_cls)

        reco_losses.prepare_losses()

        return reco_losses


# Example network implementing an entire agent by simply averaging all obvervations for all timesteps
class DynEnvFeatureExtractor(nn.Module):
    def __init__(self, features_per_object_type, feature_size, num_envs, num_rollout, num_players, num_obj_types,
                 num_time, extended_feature_cnt=0):
        super().__init__()

        # feature encoding
        self.InNet = InputLayer(features_per_object_type, feature_size, num_envs, num_players, num_obj_types, num_time)
        self.AttNet = RecurrentTemporalAttention(feature_size)

        # feature transform
        self.TransformNet = nn.Sequential(
            nn.Linear(feature_size + extended_feature_cnt, feature_size),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(feature_size),
            # nn.Linear(feature_size, feature_size),
            # nn.LeakyReLU(0.1),
        )

        self.hidden_size = feature_size
        self.LSTM = LSTMLayer(num_players, feature_size, self.hidden_size, num_envs, num_rollout)
        self.bn = nn.LayerNorm(feature_size)

    # Reset fun for lstm
    def reset(self, reset_indices=None):
        self.LSTM.reset(reset_indices)

    def detach(self):
        self.LSTM.detach()

    def forward(self, x, position=None):
        # Get embedded features
        features, objCounts = self.InNet(x)

        # Run attention
        features = self.AttNet((features, objCounts))

        # Run transformation
        if position is not None:
            features = self.TransformNet(torch.cat((features, position), dim=1))

        # Run LSTM
        features = self.LSTM(features)
        features = self.bn(features)

        # Get actions
        return features


class DynEvnEncoder(nn.Module):

    def __init__(self, feature_size, batch_size, timesteps, num_players, num_time, obs_space, obj_obs_space, reco_desc,
                 action_num, loc_feature_num):
        super().__init__()

        features_per_object_type = [flatdim(s) for s in obs_space.spaces]
        num_obj_types = len(features_per_object_type)

        self.embedder = DynEnvFeatureExtractor(features_per_object_type, feature_size, batch_size,
                                               timesteps,
                                               num_players, num_obj_types, num_time, extended_feature_cnt=action_num)

        self.predictor = nn.Linear(feature_size, loc_feature_num)

        features_per_object_type = [flatdim(s) for s in obj_obs_space.spaces]
        num_obj_types = len(features_per_object_type)

        self.objEmbedder = DynEnvFeatureExtractor(features_per_object_type, feature_size, batch_size,
                                                  timesteps,
                                                  num_players, num_obj_types, num_time, extended_feature_cnt=loc_feature_num)
        self.reconstructor = ReconNet(feature_size, reco_desc)

        self.mse = nn.MSELoss()

    def initialize(self, locInits):
        self.embedder.reset()
        self.embedder.LSTM.set_state(locInits)
        self.objEmbedder.reset()

    def compute_loc_loss(self, pos, target, faulties=None):
        # pos = pos[:len(pos)-1]

        losses = LocalizationLosses()
        if pos[0].is_cuda:
            losses.cuda()

        if faulties is None:
            faulties = torch.ones((len(pos), pos[0].shape[0])).bool()

        for p, t, f in zip(pos, target, faulties):
            loss_x = self.mse(p[f, 0], t[f, 0])
            loss_y = self.mse(p[f, 1], t[f, 1])
            loss_c = self.mse(p[f, 2], t[f, 2])
            loss_s = self.mse(p[f, 3], t[f, 3])
            loss_c_h = self.mse(p[f, 4], t[f, 4])
            loss_s_h = self.mse(p[f, 5], t[f, 5])

            corr = torch.zeros(3).cuda()

            with torch.no_grad():
                diffs = (p[:, 0] - t[:, 0]) ** 2 + (p[:, 1] - t[:, 1]) ** 2
                corr[0] = float((diffs < 0.0025).sum()) / float(len(diffs))
                corr[1] = float((diffs < 0.01).sum()) / float(len(diffs))
                corr[2] = float((diffs < 0.04).sum()) / float(len(diffs))

            losses.update_losses(loss_x, loss_y, loss_c, loss_s, loss_c_h, loss_s_h, corr)

        losses.prepare_losses(len(pos))
        return losses

    def reset(self, reset_indices=None):
        self.embedder.reset(reset_indices)
        self.objEmbedder.reset(reset_indices)

    def detach(self):
        self.embedder.detach()
        self.objEmbedder.detach()

    def get_states(self):
        s1 = self.embedder.LSTM.get_states()
        s2 = self.objEmbedder.LSTM.get_states()
        return [s1, s2]

    def set_states(self, states):
        self.embedder.LSTM.set_states(states[0])
        self.objEmbedder.LSTM.set_states(states[1])

    def forward(self, x):

        inputs, locInputs, act = x

        features = self.embedder(locInputs, act)

        pos = self.predictor(features)

        inLoc = pos.detach()

        obj_features = self.objEmbedder(inputs, inLoc)

        return features, obj_features, pos
