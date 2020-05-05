import DynEnv
from gym.spaces import MultiDiscrete, Box
import numpy as np
import pickle
import tqdm
import copy
from DynEnv.utils.utils import set_random_seeds
from time import sleep

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

    small = False

    set_random_seeds(42)
    env = DynEnv.RoboCupEnvironment(nPlayers=2, observationType=obsType, noiseType=DynEnv.NoiseType.REALISTIC, noiseMagnitude=1.0, allowHeadTurn=True, render=False)
    env.agentVisID = 0
    env.randomInit = False
    #env.deterministicTurn = True
    #env.canFall = False

    inputs = []
    locInputs = []
    outputs = []
    locOuts = []
    actInputs = []
    locInits = []
    faultys = []
    objSeens = []

    trNum = int(2**6) if small else int(2**10)
    teNum = trNum // 4
    steps = 2 #if localization else 3
    interval = 3
    faultCnt = 0
    ballS = 0
    robS = 0

    for i in tqdm.tqdm(range(trNum)):
        numPlayers = 2
        actions = generateActions(env.action_space, nPlayers=numPlayers*2, steps=steps, interval=interval)
        env.nPlayers = numPlayers

        observations = env.reset()

        state = [env.getFullState(robot) for robot in env.agents]
        inp = [[[o[0] for o in obs] for obs in observations], ]
        outp = [[s[0::2] for s in state], ]
        locO = [[s[1][:, [0, 1, 2, 3, 4, 5]] for s in state], ]
        locI = [[[o[1] for o in obs] for obs in observations], ]
        faulty = [[[o[2][0] for o in obs] for obs in observations], ]
        objSeen = [[[o[2][1:] for o in obs] for obs in observations], ]
        initLoc = locO[0]

        for action in actions:
            observations, _, _, info = env.step(action)
            state = info['Recon States']
            inp.append([[o[0] for o in obs] for obs in observations])
            locI.append([[o[1] for o in obs] for obs in observations])
            locO.append([s[1][:, [0,1,2,3,4,5]] for s in state])
            faulty.append([[o[2][0] for o in obs] for obs in observations])
            objSeen.append([[o[2][1:] for o in obs] for obs in observations])
            outp.append([s[0::2] for s in state])

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
        actions = [np.zeros((4,4)),] + actions
        actInputs.append(actions)
        faultys.append(faulty)
        objSeens.append(objSeen)
        locInits.append(initLoc)
        faultCnt += (np.array(faulty).sum(axis=1) == 0).sum()
        robS += np.array([[[[s for s in rob[0]] for rob in sight] for sight in time] for time in objSeen]).any(axis=1).mean()
        ballS += np.array([[[rob[1] for rob in sight] for sight in time] for time in objSeen]).any(axis=1).mean()

    trDataNum = len(inputs)
    trainData = [locInputs, inputs, locOuts, outputs, actInputs, faultys, objSeens, locInits]
    print("Faulty: ", faultCnt, "Balls: ", ballS/len(inputs)*100, "Robots: ", robS/len(inputs)*100)
    sleep(0.5)

    inputs = []
    locInputs = []
    outputs = []
    locOuts = []
    actInputs = []
    locInits = []
    faultys = []
    objSeens = []
    faultCnt = 0
    ballS = 0
    robS = 0

    for i in tqdm.tqdm(range(teNum)):
        numPlayers = 2
        actions = generateActions(env.action_space, nPlayers=numPlayers * 2, steps=steps)
        env.nPlayers = numPlayers


        observations = env.reset()

        state = [env.getFullState(robot) for robot in env.agents]
        inp = [[[o[0] for o in obs] for obs in observations],]
        outp = [[s[0::2] for s in state],]
        locO = [[s[1][:, [0,1,2,3,4,5]] for s in state],]
        locI = [[[o[1] for o in obs] for obs in observations],]
        faulty = [[[o[2][0] for o in obs] for obs in observations], ]
        objSeen = [[[o[2][1:] for o in obs] for obs in observations], ]
        initLoc = locO[0]

        for action in actions:
            observations, _, _, info = env.step(action)
            state = info['Recon States']
            inp.append([[o[0] for o in obs] for obs in observations])
            locI.append([[o[1] for o in obs] for obs in observations])
            locO.append([s[1][:, [0,1,2,3,4,5]] for s in state])
            faulty.append([[o[2][0] for o in obs] for obs in observations])
            objSeen.append([[o[2][1:] for o in obs] for obs in observations])
            outp.append([s[0::2] for s in state])

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
        actions = [np.zeros((4,4)),] + actions
        actInputs.append(actions)
        locInits.append(initLoc)
        faultys.append(faulty)
        objSeens.append(objSeen)
        faultCnt += (np.array(faulty).sum(axis=1) == 0).sum()
        robS += np.array([[[[s for s in rob[0]] for rob in sight] for sight in time] for time in objSeen]).any(axis=1).mean()
        ballS += np.array([[[rob[1] for rob in sight] for sight in time] for time in objSeen]).any(axis=1).mean()

    teDataNum = len(inputs)
    testData = [locInputs, inputs, locOuts, outputs, actInputs, faultys, objSeens, locInits]
    print("Faulty: ", faultCnt, "Balls: ", ballS/len(inputs)*100, "Robots: ", robS/len(inputs)*100)


    print("%d training and %d test datapoints generated." % (trDataNum, teDataNum))
    print("Saving")

    baseName = "data/roboSmall" if small else "data/robo"
    file = open(baseName + "Train.pickle","wb")
    pickle.dump(trainData,file)

    file = open(baseName + "Test.pickle","wb")
    pickle.dump(testData,file)
    print("Done")
