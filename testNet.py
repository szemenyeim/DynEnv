from models import *
import DynEnv
import torch

def doRoboCup():
    nPlayers = 5
    env = DynEnv.RoboCupEnvironment(nPlayers=nPlayers, render=False, observationType=DynEnv.ObservationType.Partial,
                                    noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=2)
    observations = env.setRandomSeed(42)

    batch, action = env.getActionSize()
    feature = 128
    Net = ActionLayer(feature,action).cuda()

    while True:

        inputs = torch.randn(batch,feature).cuda()
        actions = Net(inputs)

        actions = torch.stack([torch.argmax(actions[0],dim=1).float(),
                            torch.argmax(actions[1],dim=1).float(),
                            torch.argmax(actions[2],dim=1).float(),
                            actions[3].squeeze()],dim=1)


        #action = Net(observations)
        state,observations,trewards,rrewads,finished = env.step(actions.detach().cpu())
        if finished:
            observations = env.reset()

def doDrive():
    nPlayers = 5
    env = DynEnv.DrivingEnvironment(nPlayers=nPlayers, render=True, observationType=DynEnv.ObservationType.Partial,
                                    noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=2)


    batch, action = env.getActionSize()
    feature = 128
    Net = ActionLayer(feature,action).cuda()

    while True:

        inputs = torch.randn(batch,feature).cuda()
        actions = Net(inputs)[0]

        #action = Net(observations)
        state, observations, trewards, rrewads, finished = env.step(actions.detach().cpu())
        if finished:
            observations = env.reset()

if __name__ == '__main__':
    drive = True


    if drive:
        doDrive()
    else:
        doRoboCup()