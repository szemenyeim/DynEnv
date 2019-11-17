import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

flatten = lambda l: [item for sublist in l for item in sublist]


# Concatenate tensors and pad them to size
def catAndPad(list, size):
    t = torch.cat(list)
    return F.pad(t, pad=[0, 0, 0, size - t.shape[0]], mode='constant', value=0)


# Convert maxindices to a binary mask
def createMask(counts,max):
    mask = torch.zeros((counts.shape[0],max)).bool()
    for i,count in enumerate(counts):
        mask[i, count:] = True
    return mask


# Helper object to convert indices in a for loop quickly
class Indexer(object):
    def __init__(self, arrNum):
        self.prev = np.zeros(arrNum).astype('int64')

    def getRange(self, counts, time, unit, arrIdx):
        if time == 0 and unit == 0:
            self.prev[arrIdx] = 0
        count = counts[time][unit]
        self.prev[arrIdx] += count
        return range(self.prev[arrIdx] - count, self.prev[arrIdx])


# Outputs a certain type of action
class ActionBlock(nn.Module):
    def __init__(self, features, actions, type, means=None, scale=None):
        super(ActionBlock, self).__init__()

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
        self.activation = nn.Softmax(dim=1) if type == 'cat' else nn.Sigmoid()

    # Put means and std on the correct device when .cuda() or .cpu() is called
    def _apply(self, fn):
        super(ActionBlock, self)._apply(fn)
        if self.means is not None:
            self.means = fn(self.means)
            self.scale = fn(self.scale)
        return self

    # Forward
    def forward(self, x):

        # Forward
        x = self.activation(self.Layer(x))

        # Optional scaling
        if self.means is not None:
            x = (x - 0.5) * self.scale + self.means

        return x


# Complete action layer for multiple action groups
class ActionLayer(nn.Module):
    def __init__(self, features, actions):
        super(ActionLayer, self).__init__()

        # Create action groups
        self.blocks = nn.ModuleList(
            [ActionBlock(features, action[1], action[0], action[2], action[3]) for action in actions])

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
    def __init__(self, inputs, features):
        super(InputLayer, self).__init__()

        # To be added in the future
        if len(inputs) != 4:
            raise Exception("Image and Full observation types are not yet supported")

        # Number of features of different input object types
        inputNums = inputs[-1]

        # Basic params
        self.nTime = inputs[0]
        self.nPlayers = inputs[1]
        self.nObjects = inputs[2]

        # Helper class for arranging tensor
        self.indexer = Indexer(self.nObjects)

        # Create embedding blocks for them
        self.blocks = nn.ModuleList([EmbedBlock(input, features) for input in inputNums])

    def forward(self, x):
        # Get device
        device = next(self.parameters()).device

        # Object counts [type x timeStep x nPlayer]
        counts = [[[len(sightings[i]) for sightings in time] for time in x] for i in range(self.nObjects)]
        objCounts = torch.Tensor(
            [[sum([counts[k][i][j] for k in range(self.nObjects)]) for j in range(self.nPlayers)] for i in
             range(self.nTime)]).long()
        maxCount = torch.max(objCounts).item()

        # Object arrays [all objects of the same type]
        ins = [np.stack(flatten([flatten([sightings[i] for sightings in time if len(sightings[i])]) for time in x])) for
               i in range(self.nObjects)]

        # Call embedding block for all object types
        outs = [block(torch.Tensor(obj).to(device)) for block, obj in zip(self.blocks, ins)]

        # Arrange objects in tensor [TimeSteps x maxObjCnt x nPlayers x featureCnt] using padding
        outs = torch.stack(
                [torch.stack(
                [catAndPad([out[self.indexer.getRange(counts[k],i,j,k)]
                                                    for k,out in enumerate(outs)],maxCount)
                                         for j in range(self.nPlayers)])
                            for i in range(self.nTime)]).permute(0,2,1,3)

        return outs,objCounts


# Layer for reducing timeStep and Objects dimension via attention
class AttentionLayer(nn.Module):
    def __init__(self,features,num_heads=1):
        super(AttentionLayer,self).__init__()

        # objAtt implements self attention between objects seen in the same timestep
        self.objAtt = nn.MultiheadAttention(features,num_heads)

        # Temp att attends between sightings at different timesteps
        self.tempAtt = nn.MultiheadAttention(features,num_heads)

    def forward(self,x):

        # Get device
        device = next(self.parameters()).device

        # create masks
        tensor, objCounts = x
        maxNum = tensor.shape[1]
        masks = [createMask(counts,maxNum).to(device) for counts in objCounts]

        # Run self-attention
        attObj = [self.objAtt(objs,objs,objs,mask)[0] for objs,mask in zip(tensor,masks)]

        # Run temporal attention
        finalAtt = attObj[0]
        finalMask = masks[0]
        for i in range(0,len(attObj)-1):
            finalAtt = self.tempAtt(attObj[i+1],finalAtt,finalAtt,finalMask)[0]
            finalMask = masks[i+1]

        # Mask out final attention results
        finalMask = finalMask.permute(1,0)
        finalAtt[finalMask] = 0

        # Masked averaging
        summed = torch.sum(finalAtt,0)
        lens = torch.sum(torch.logical_not(finalMask),0).float().unsqueeze(1)

        return summed.div(lens)


# LSTM Layer
class LSTMLayer(nn.Module):
    def __init__(self,nPlayers,feature,hidden):
        super(LSTMLayer,self).__init__()

        # Params
        self.nPlayers = nPlayers
        self.feature = feature
        self.hidden = hidden

        # Init LSTM cell
        self.reset()
        self.cell = nn.LSTMCell(feature,hidden)

    # Reset inner state
    def reset(self):
        self.h = torch.zeros((self.nPlayers,self.hidden))
        self.c = torch.zeros((self.nPlayers,self.hidden))

    # Put means and std on the correct device when .cuda() or .cpu() is called
    def _apply(self, fn):
        super(LSTMLayer, self)._apply(fn)

        self.h = fn(self.h)
        self.c = fn(self.c)

        return self

    # Forward
    def forward(self, x):

        self.h, self.c = self.cell(x,(self.h,self.c))

        return self.h


# Example network implementing an entire agent by simply averaging all obvervations for all timesteps
class TestNet(nn.Module):
    def __init__(self,inputs,action,feature):
        super(TestNet,self).__init__()

        nPlayers = inputs[1]

        self.InNet = InputLayer(inputs,feature)
        self.AttNet = AttentionLayer(feature)
        self.LSTM = LSTMLayer(nPlayers,feature,feature*2)
        self.OutNet = ActionLayer(feature*2,action)

    # Reset fun for lstm
    def reset(self):
        self.LSTM.reset()

    def forward(self, x):
        # Get embedded features
        features,objCounts = self.InNet(x)

        # Run attention
        features = self.AttNet((features,objCounts))

        # Run LSTM
        features = self.LSTM(features)

        # Get actions
        return self.OutNet(features)
