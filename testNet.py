from models import *
import DynEnv
import torch

def doRoboCup():

    # Create env
    isCuda =torch.cuda.is_available()
    nPlayers = 5
    env = DynEnv.RoboCupEnvironment(nPlayers=nPlayers, render=True, observationType=DynEnv.ObservationType.Partial,
                                    noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=0)

    # Set random seed and get inital observation
    observations = env.setRandomSeed(42)

    # Get network
    batch, action = env.getActionSize()
    inputs = env.getObservationSize()
    feature = 128
    Net = TestNet(inputs,action,feature)
    if isCuda:
        Net = Net.cuda()

    while True:

        # Forward
        actions = Net(observations)

        # Sample categorical actions deterministically and stack them
        actions = torch.stack([torch.argmax(actions[0],dim=1).float(),
                            torch.argmax(actions[1],dim=1).float(),
                            torch.argmax(actions[2],dim=1).float(),
                            actions[3].squeeze()],dim=1)

        # Step
        state,observations,trewards,rrewads,finished = env.step(actions.detach().cpu())

        # Finished
        if finished:
            observations = env.reset()
            Net.reset()

def doDrive():

    # Create env
    isCuda = torch.cuda.is_available()
    nPlayers = 5
    env = DynEnv.DrivingEnvironment(nPlayers=nPlayers, render=True, observationType=DynEnv.ObservationType.Partial,
                                    noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=2)

    # Set random seed and get inital observation
    observations = env.setRandomSeed(42)

    # Get network
    batch, action = env.getActionSize()
    inputs = env.getObservationSize()
    feature = 128
    Net = TestNet(inputs, action, feature)
    if isCuda:
        Net = Net.cuda()

    while True:

        # Forward
        actions = Net(observations)[0]

        # Step
        state, observations, trewards, rrewads, finished = env.step(actions.detach().cpu())

        # Finished
        if finished:
            observations = env.reset()
            Net.reset()

if __name__ == '__main__':

    drive = True

    if drive:
        doDrive()
    else:
        doRoboCup()