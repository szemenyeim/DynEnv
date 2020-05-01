import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunkoptions

pymunkoptions.options["debug"] = False
from DynEnv import *
from DynEnv.models.agent import ICMAgent
from DynEnv.models import DynEvnEncoder
from DynEnv.models.models import ReconLosses, LocalizationLosses
from DynEnv.utils.utils import set_random_seeds, AttentionTarget, AttentionType, flatten
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import progressbar
from gym.spaces import flatdim
import torch.nn as nn

def train(epoch):

    losses = LocalizationLosses()
    losses.cuda()

    reconLosses = ReconLosses(num_classes=2, num_thresh=3)
    reconLosses.cuda()

    net.train()

    bar = progressbar.ProgressBar(0, len(trLoader))

    for i, (ind, _) in enumerate(trLoader):

        locInputs = trainData[0][ind].T
        inputs = trainData[1][ind].T
        locTargets = trainData[2][ind].T
        targets = trainData[3][ind].T
        actions = trainData[4][ind].T
        faulties = trainFaults[ind].permute(1,0,2,3).all(dim=2).view((timesteps, -1))
        locInits = torch.tensor(trainInitLocs[ind])
        locInits += (torch.randn(locInits.shape))/50.0
        locInits = locInits.view(-1, locInits.shape[-1])
        #locs = torch.zeros(locInits.shape[0], feature_size)
        #locs[:,:locInits.shape[1]] = locInits

        net.initialize(locInits)

        #rec_optimizer.zero_grad()
        optimizer.zero_grad()

        poses = []
        locTars = []

        bar.update(i)

        for j, (lI,lT,act,I) in enumerate(zip(locInputs, locTargets, actions, inputs)):
            act = torch.tensor(flatten(list(act))).float().cuda().squeeze()

            features, obj_features, pos = net((I, lI, act), True)

            lT = torch.tensor(flatten(list(lT))).cuda().squeeze()

            poses.append(pos)
            locTars.append(lT)

        loss = net.compute_loc_loss(poses, locTars)
        recLosses = net.reconstructor(obj_features, targets[-1])

        derLoss = loss.loss #* locFactor + recLosses.loss
        derLoss.backward()

        optimizer.step()

        loss.detach_loss()
        losses += loss
        recLosses.detach_loss()
        reconLosses += recLosses

    bar.finish()

    losses.div(len(trLoader))
    reconLosses.div(len(trLoader))
    losses.finalize_corr()
    reconLosses.compute_APs()

    print("[Train Epoch %d/%d]" % (epoch + 1, epochNum), flush=True)
    print(losses, flush=True)
    print(reconLosses, flush=True)

def val(epoch):

    losses = LocalizationLosses()
    losses.cuda()

    reconLosses = ReconLosses(num_classes=2, num_thresh=3)
    reconLosses.cuda()

    net.eval()

    bar = progressbar.ProgressBar(0, len(teLoader))

    for i, (ind, _) in enumerate(teLoader):

        locInputs = testData[0][ind].T
        inputs = testData[1][ind].T
        locTargets = testData[2][ind].T
        targets = testData[3][ind].T
        actions = testData[4][ind].T
        faulties = testFaults[ind].permute(1,0,2,3).all(dim=2).view((timesteps, -1))
        locInits = torch.tensor(testInitLocs[ind])
        locInits += (torch.randn(locInits.shape))/50.0
        locInits = locInits.view(-1, locInits.shape[-1])
        #locs = torch.zeros(locInits.shape[0], feature_size)
        #locs[:,:locInits.shape[1]] = locInits

        net.initialize(locInits)

        poses = []
        locTars = []

        bar.update(i)

        for j, (lI, lT, act, I) in enumerate(zip(locInputs, locTargets, actions, inputs)):
            act = torch.tensor(flatten(list(act))).float().cuda().squeeze()

            features, obj_features, pos = net((I, lI, act), True)

            lT = torch.tensor(flatten(list(lT))).cuda().squeeze()

            poses.append(pos)
            locTars.append(lT)

        loss = net.compute_loc_loss(poses, locTars)

        recLosses = net.reconstructor(obj_features, targets[-1])

        loss.detach_loss()
        losses += loss

        recLosses.detach_loss()
        reconLosses += recLosses

    bar.finish()

    losses.div(len(teLoader))
    reconLosses.div(len(teLoader))
    losses.finalize_corr()
    reconLosses.compute_APs()

    print("[Test Epoch %d/%d]" % (epoch + 1, epochNum), flush=True)
    print(losses, flush=True)
    print(reconLosses, flush=True)

    avg = losses.corr.mean()
    avg += reconLosses.APs.mean()

    return avg/2


if __name__ == '__main__':

    # seeds set (all but env)
    set_random_seeds(42)

    small = False
    baseName = "roboSmall" if small else "robo"

    # constants
    feature_size = 64
    attn_target = AttentionTarget.ICM_LOSS
    attn_type = AttentionType.SINGLE_ATTENTION

    # Localization relative weight
    locFactor = 1.0

    # env
    env = RoboCupEnvironment(1, observationType=ObservationType.PARTIAL)
    action_size = env.action_space
    obs_space = env.observation_space
    loc_obs_space = obs_space[1]
    obj_obs_space = obs_space[0]

    reco_desc = env.recoDescriptor

    batch_size = 8 if small else 16
    num_players = 4
    timesteps = 7 #if localization else 9
    epochNum = 30
    num_time = 5

    net = DynEvnEncoder(feature_size, batch_size, timesteps, num_players, num_time, loc_obs_space, obj_obs_space, reco_desc).cuda()

    '''if not localization:
        suffix = "Loc.pth"
        net.load_state_dict(torch.load("models/net" + suffix))'''

    #params = list(net.embedder.parameters()) + list(net.predictor.parameters())

    loc_lr = 1e-3 #if localization else 1e-5
    #rec_lr = 1e-3

    optimizer = torch.optim.Adam(net.parameters(), lr=loc_lr, weight_decay=0)
    sceduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochNum, loc_lr/10)

    '''params = list(net.objEmbedder.parameters()) + list(net.reconstructor.parameters())

    rec_optimizer = torch.optim.Adam(params, lr=rec_lr, weight_decay=0)
    rec_sceduler = torch.optim.lr_scheduler.CosineAnnealingLR(rec_optimizer, epochNum, rec_lr/10)'''

    file = open(baseName + 'Train.pickle', 'rb')
    trainData = pickle.load(file)
    trainInitLocs = np.array(trainData[-1])
    trainFaults = ~torch.tensor(trainData[-2])
    trainData = np.array(trainData[:-2])
    file = open(baseName + 'Test.pickle', 'rb')
    testData = pickle.load(file)
    testInitLocs = np.array(testData[-1])
    testFaults = ~torch.tensor(testData[-2])
    testData = np.array(testData[:-2])

    trTens = torch.arange(len(trainData[0]))
    teTens = torch.arange(len(testData[0]))

    trSet = TensorDataset(trTens,trTens)
    teSet = TensorDataset(teTens,teTens)

    trLoader = DataLoader(trSet,batch_size=batch_size,shuffle=True)
    teLoader = DataLoader(teSet,batch_size=batch_size,shuffle=False)

    bestAvg = 0

    for epoch in range(epochNum):

        train(epoch)
        avg = val(epoch)
        sceduler.step()
        #rec_sceduler.step()

        if avg > bestAvg and not small:
            bestAvg = avg
            print("Saving best model: %.2f" % avg, flush=True)
            suffix = "Rec.pth"
            torch.save(net.state_dict(),"models/net" + suffix)


    print("Best: ", bestAvg)