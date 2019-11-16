from models import *
import DynEnv
import torch

def doRoboCup():
    nPlayers = 5
    env = DynEnv.RoboCupEnvironment(nPlayers=nPlayers, render=True, observationType=DynEnv.ObservationType.Partial,
                                    noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=2)
    observations = env.setRandomSeed(42)

    batch, action = env.getActionSize()
    inputs = env.getObservationSize()
    feature = 128
    Net = TestNet(inputs,action,feature).cuda()

    while True:

        actions = Net(observations)

        actions = torch.stack([torch.argmax(actions[0],dim=1).float(),
                            torch.argmax(actions[1],dim=1).float(),
                            torch.argmax(actions[2],dim=1).float(),
                            actions[3].squeeze()],dim=1)


        state,observations,trewards,rrewads,finished = env.step(actions.detach().cpu())
        if finished:
            observations = env.reset()

def doDrive():
    nPlayers = 5
    env = DynEnv.DrivingEnvironment(nPlayers=nPlayers, render=True, observationType=DynEnv.ObservationType.Partial,
                                    noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=2)

    observations = env.setRandomSeed(42)

    batch, action = env.getActionSize()
    inputs = env.getObservationSize()
    feature = 128
    Net = TestNet(inputs,action,feature).cuda()

    while True:

        actions = Net(observations)[0]
        state, observations, trewards, rrewads, finished = env.step(actions.detach().cpu())
        if finished:
            observations = env.reset()

if __name__ == '__main__':
    drive = True


    if drive:
        doDrive()
    else:
        doRoboCup()