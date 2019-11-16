import torch
import torch.nn as nn

class ActionBlock(nn.Module):
    def __init__(self,features,actions,type,means=None,scale=None):
        super(ActionBlock,self).__init__()

        self.means = None
        self.scale = None

        if means is not None:
            self.means = torch.Tensor(means)
            self.scale = torch.Tensor(scale)
            assert (len(means) == len(scale) and len(means) == actions)

        self.Layer = nn.Linear(features,actions)
        self.activation = nn.Softmax(dim=1) if type == 'cat' else nn.Sigmoid()

    def _apply(self, fn):
        super(ActionBlock, self)._apply(fn)
        if self.means is not None:
            self.means = fn(self.means)
            self.scale = fn(self.scale)
        return self

    def forward(self, x):
        x = self.activation(self.Layer(x))
        if self.means is not None:
            x = (x-0.5)*self.scale + self.means
        return x

class ActionLayer(nn.Module):
    def __init__(self,features,actions):
        super(ActionLayer,self).__init__()

        self.blocks = nn.ModuleList([ActionBlock(features,action[1],action[0],action[2],action[3]) for action in actions])

    def forward(self, x):
        outs = [block(x) for block in self.blocks]
        return outs

class EmbedBlock(nn.Module):
    def __init__(self,inputs,features):
        super(EmbedBlock,self).__init__()

        self.Layer = nn.Linear(inputs,features)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self,x):
        if x.dim() == 1:
            if x.shape[0] == 0:
                return None
            x = x.unsqueeze(0)
        return self.relu(self.Layer(x))

class InputLayer(nn.Module):
    def __init__(self,inputs,features):
        super(InputLayer,self).__init__()

        if len(inputs) != 4:
            raise Exception("Image and Full observation types are not yet supported")

        inputNums = inputs[-1]

        self.blocks = nn.ModuleList([EmbedBlock(input,features) for input in inputNums])

    def forward(self, x):
        device = next(self.parameters()).device

        outs = [[[block(torch.Tensor(obj).to(device)) for block,obj in zip(self.blocks,player)] for player in time] for time in x]

        outs = [[torch.cat([t for t in player if t is not None],dim=0) for player in time] for time in outs]

        # Change list from [time x nPlayer x objNum x features] to [nPlayers x time x objNum x features]
        outs = list(map(list, zip(*outs)))

        return outs

class TestNet(nn.Module):
    def __init__(self,inputs,action,feature):
        super(TestNet,self).__init__()

        self.InNet = InputLayer(inputs,feature)
        self.OutNet = ActionLayer(feature,action)

    def forward(self, x):
        features = self.InNet(x)
        features = [sum([torch.mean(f,dim=0) for f in feature])/len(feature) for feature in features]
        features = torch.stack(features)
        actions = self.OutNet(features)
        return actions