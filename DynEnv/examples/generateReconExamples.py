import DynEnv
from gym.spaces import MultiDiscrete, Box
import numpy as np
import pickle
import tqdm
import copy
from DynEnv.utils.utils import set_random_seeds

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
                    actions.append((np.random.rand(nPlayers) * (a_max - a_min) + a_min)/3)

        actions = np.array(actions).T
        for j in range(interval):
            fullActions += [copy.deepcopy(actions),]
    return fullActions

if __name__ == '__main__':

    obsType = DynEnv.ObservationType.PARTIAL

    localization = False

    set_random_seeds(42)
    env = DynEnv.RoboCupEnvironment(nPlayers=2, observationType=obsType, noiseType=DynEnv.NoiseType.REALISTIC, noiseMagnitude=1.0, allowHeadTurn=True, render=False)
    env.agentVisID = 0
    env.randomInit = True
    env.deterministicTurn = not localization
    env.canFall = False

    inputs = []
    locInputs = []
    outputs = []
    locOuts = []
    actInputs = []
    locInits = []

    trNum = int(2**10) if localization else int(2**6)
    teNum = trNum // 4
    steps = 2 #if localization else 3
    interval = 3

    for i in tqdm.tqdm(range(trNum)):
        numPlayers = 2
        actions = generateActions(env.action_space, nPlayers=numPlayers*2, steps=steps, interval=interval)
        env.nPlayers = numPlayers

        while True:
            faulty = False

            env.reset()
            inp = []
            outp = []
            locO = []
            locI = []
            act = []

            initLoc = [env.getFullState(robot)[1][:, [0,1,4,5]] for robot in env.agents]

            for action in actions:
                observations, _, _, info = env.step(action)
                state = info['Recon States']
                faulty = info['Faulty']
                if faulty:
                    break
                inp.append([[o[0] for o in obs] for obs in observations])
                locI.append([[o[1] for o in obs] for obs in observations])
                locO.append([s[1][:, [0,1,4,5]] for s in state])
                outp.append([s[0::2] for s in state])

            if not faulty:
                break
            print("Faulty, retrying")

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
        locInits.append(initLoc)

    trDataNum = len(inputs)
    trainData = [locInputs, inputs, locOuts, outputs, actInputs, locInits]

    inputs = []
    locInputs = []
    outputs = []
    locOuts = []
    actInputs = []
    locInits = []

    for i in tqdm.tqdm(range(teNum)):
        numPlayers = 2
        actions = generateActions(env.action_space, nPlayers=numPlayers * 2, steps=steps)
        env.nPlayers = numPlayers

        while True:
            faulty = False

            env.reset()
            inp = []
            outp = []
            locO = []
            locI = []
            initLoc = [env.getFullState(robot)[1][:, [0,1,4,5]] for robot in env.agents]

            for action in actions:
                observations, _, _, info = env.step(action)
                state = info['Recon States']
                faulty = info['Faulty']
                if faulty:
                    break
                inp.append([[o[0] for o in obs] for obs in observations])
                locI.append([[o[1] for o in obs] for obs in observations])
                locO.append([s[1][:, [0,1,4,5]] for s in state])
                outp.append([s[0::2] for s in state])

            if not faulty:
                break
            print("Faulty, retrying")

        inputs.append(inp)
        outputs.append(outp)
        locOuts.append(locO)
        locInputs.append(locI)
        actInputs.append(actions)
        locInits.append(initLoc)

    teDataNum = len(inputs)
    testData = [locInputs, inputs, locOuts, outputs, actInputs, locInits]

    print("%d training and %d test datapoints generated." % (trDataNum, teDataNum))
    print("Saving")

    baseName = "roboLoc" if localization else "roboRec"
    file = open(baseName + "Train.pickle","wb")
    pickle.dump(trainData,file)

    file = open(baseName + "Test.pickle","wb")
    pickle.dump(testData,file)
    print("Done")
