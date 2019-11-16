import torch
import torch.nn as nn

# Outputs a certain type of action
class ActionBlock(nn.Module):
    def __init__(self,features,actions,type,means=None,scale=None):
        super(ActionBlock,self).__init__()

        # Initialize
        self.means = None
        self.scale = None

        # For continous actions a desired interval can be given [mean-range:mean+range] (otherwise [0:1])
        if means is not None:
            self.means = torch.Tensor(means)
            self.scale = torch.Tensor(scale)
            assert (len(means) == len(scale) and len(means) == actions)

        # Create layers
        self.Layer = nn.Linear(features,actions)
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
            x = (x-0.5)*self.scale + self.means

        return x

# Complete action layer for multiple action groups
class ActionLayer(nn.Module):
    def __init__(self,features,actions):
        super(ActionLayer,self).__init__()

        # Create action groups
        self.blocks = nn.ModuleList([ActionBlock(features,action[1],action[0],action[2],action[3]) for action in actions])

    # Return list of actions
    def forward(self, x):
        outs = [block(x) for block in self.blocks]
        return outs

# Simple embedding block for a single object type
class EmbedBlock(nn.Module):
    def __init__(self,inputs,features):
        super(EmbedBlock,self).__init__()

        self.Layer = nn.Linear(inputs,features)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self,x):

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
    def __init__(self,inputs,features):
        super(InputLayer,self).__init__()

        # To be added in the future
        if len(inputs) != 4:
            raise Exception("Image and Full observation types are not yet supported")

        # Number of features of different input object types
        inputNums = inputs[-1]

        # Create embedding blocks for them
        self.blocks = nn.ModuleList([EmbedBlock(input,features) for input in inputNums])

    def forward(self, x):

        # Get device
        device = next(self.parameters()).device

        # Call embedding block for all objects seen by all players in every observation timestep (sadly this can't be batched)
        outs = [[[block(torch.Tensor(obj).to(device)) for block,obj in zip(self.blocks,player)] for player in time] for time in x]

        # Concatenate object types seen by a signle player in a single timestep (these now have the same number of features)
        outs = [[torch.cat([t for t in player if t is not None],dim=0) for player in time] for time in outs]

        # Change list from [time x nPlayer x objNum x features] to [nPlayers x time x objNum x features]
        outs = list(map(list, zip(*outs)))

        return outs

# Example network implementing an entire agent by simply averaging all obvervations for all timesteps
class TestNet(nn.Module):
    def __init__(self,inputs,action,feature):
        super(TestNet,self).__init__()

        self.InNet = InputLayer(inputs,feature)
        self.OutNet = ActionLayer(feature,action)

    def forward(self, x):

        # Get embedded features
        features = self.InNet(x)

        # First, average objects seen in the same timestep, then average timesteps
        features = [sum([torch.mean(f,dim=0) for f in feature])/len(feature) for feature in features]

        # Convert list containing the internal feature of every robot to tensor (enable batching)
        features = torch.stack(features)

        # Get actions
        return self.OutNet(features)