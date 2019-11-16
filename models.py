import torch
import torch.nn as nn

class ActionBlock(nn.Module):
    def __init__(self,features,actions,type,means=None,scale=None):
        super(ActionBlock,self).__init__()

        self.means = None
        self.scale = None

        if means is not None:
            self.means = torch.Tensor(means).cuda()
            self.scale = torch.Tensor(scale).cuda()
            assert (len(means) == len(scale) and len(means) == actions)

        self.Layer = nn.Linear(features,actions)
        self.activation = nn.Softmax(dim=1) if type == 'cat' else nn.Sigmoid()

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
