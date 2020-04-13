import DynEnv
from gym.spaces import MultiDiscrete, Box
import numpy as np
import pickle
import tqdm
import copy

def generateActions(actionDefs,nPlayers,steps,interval = 3):
    fullActions = []
    for i in range(steps):
        actions = []
        for d in actionDefs:
            if type(d) == MultiDiscrete:
                for a_num in d.nvec:
                    actions.append(np.random.randint(0,a_num, nPlayers))
            elif type(d) == Box:
                for a_min, a_max in zip(d.low, d.high):
                    actions.append((np.random.rand(nPlayers) * (a_max - a_min) + a_min)/2)

        actions = np.array(actions).T
        for j in range(interval):
            fullActions += [copy.deepcopy(actions),]
    return fullActions

if __name__ == '__main__':

    obsType = DynEnv.ObservationType.PARTIAL

    env = DynEnv.RoboCupEnvironment(nPlayers=2, observationType=obsType, noiseType=DynEnv.NoiseType.REALISTIC, noiseMagnitude=0.0, allowHeadTurn=False)
    env.randomInit = True

    inputs = []
    locInputs = []
    outputs = []
    locOuts = []
    actInputs = []

    trNum = int(2**6)
    teNum = trNum // 4
    steps = 2

    for i in tqdm.tqdm(range(trNum)):
        numPlayers = 2
        actions = generateActions(env.action_space, nPlayers=numPlayers*2, steps=steps)
        env.nPlayers = numPlayers

        env.reset()
        inp = []
        outp = []
        locO = []
        locI = []
        act = []

        for action in actions:
            observations, _, _, info = env.step(action)
            state = info['Recon States']
            inp.append([[o[0] for o in obs] for obs in observations])
            locI.append([[o[1] for o in obs] for obs in observations])
            locO.append([s[1][:, [0,1,4,5]] for s in state])
            outp.append(state)

        for action in actions:
            action[:,2] = np.copy(action[:,0])
            turns = action[:,1]
            turns[turns==2] = -1
            forwards = action[:,0]
            forwards[forwards<3] = 0
            forwards[forwards == 3] = 1
            forwards[forwards == 4] = -1
            sides = action[:,2]
            sides[sides > 2] = 0
            sides[sides == 2] = 1
            sides[sides == 1] = -1
        inputs.append(inp)
        outputs.append(outp)
        locOuts.append(locO)
        locInputs.append(locI)
        actInputs.append(actions)

    trDataNum = len(inputs)
    trainData = [locInputs, inputs, locOuts, outputs, actInputs]

    inputs = []
    locInputs = []
    outputs = []
    locOuts = []
    actInputs = []

    for i in tqdm.tqdm(range(teNum)):
        numPlayers = 2
        actions = generateActions(env.action_space, nPlayers=numPlayers * 2, steps=steps)
        env.nPlayers = numPlayers

        env.reset()
        inp = []
        outp = []
        locO = []
        locI = []

        for action in actions:
            observations, _, _, info = env.step(action)
            state = info['Recon States']
            inp.append([[o[0] for o in obs] for obs in observations])
            locI.append([[o[1] for o in obs] for obs in observations])
            locO.append([s[1][:, [0,1,4,5]] for s in state])
            outp.append(state)

        inputs.append(inp)
        outputs.append(outp)
        locOuts.append(locO)
        locInputs.append(locI)
        actInputs.append(actions)

    teDataNum = len(inputs)
    testData = [locInputs, inputs, locOuts, outputs, actInputs]

    print("%d training and %d test datapoints generated." % (trDataNum, teDataNum))
    print("Saving")

    file = open("roboFullTrain.pickle","wb")
    pickle.dump(trainData,file)

    file = open("roboFullTest.pickle","wb")
    pickle.dump(testData,file)
    print("Done")
