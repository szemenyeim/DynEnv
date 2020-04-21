import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunkoptions

pymunkoptions.options["debug"] = False
from DynEnv import *
from DynEnv.models.agent import ICMAgent
from DynEnv.models import DynEnvFeatureExtractor
from DynEnv.utils.utils import set_random_seeds, AttentionTarget, AttentionType, flatten
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import progressbar
from gym.spaces import flatdim
import torch.nn as nn

def train(epoch):

    losses = np.zeros(9)

    embedder.train()

    bar = progressbar.ProgressBar(0, len(trLoader)*timesteps, redirect_stdout=False)

    corr = [0,0,0]

    for i, (ind, _) in enumerate(trLoader):

        locInputs = trainData[0][ind].T
        inputs = trainData[1][ind].T
        locTargets = trainData[2][ind].T
        targets = trainData[3][ind].T
        actions = trainData[4][ind].T

        embedder.reset()

        for j, (lI,lT,act) in enumerate(zip(locInputs,locTargets,actions)):

            act = torch.tensor(flatten(list(act))).float().cuda().squeeze()

            optimizer.zero_grad()

            features = torch.cat((embedder(lI),act), dim=1)

            pos = predictor(features)

            lT = torch.tensor(flatten(list(lT))).cuda().squeeze()

            loss_x = criterion(pos[:,0],lT[:,0])
            loss_y = criterion(pos[:,1],lT[:,1])
            loss_c = criterion(pos[:,2],lT[:,2])
            loss_s = criterion(pos[:,3],lT[:,3])

            with torch.no_grad():
                diffs = (pos[:,0]-lT[:,0])**2 + (pos[:,1]-lT[:,1])**2
                corr[0] += (diffs < 0.0025).sum()
                corr[1] += (diffs < 0.01).sum()
                corr[2] += (diffs < 0.04).sum()

            loss = loss_x + loss_y + loss_c + loss_s

            loss.backward(retain_graph=True)
            optimizer.step()

            losses[0] += loss_x.item()/len(locInputs)
            losses[1] += loss_y.item()/len(locInputs)
            losses[2] += loss_c.item()/len(locInputs)
            losses[3] += loss_s.item()/len(locInputs)
            losses[4] += loss.item()/len(locInputs)

            bar.update(i*timesteps + j)


    bar.finish()

    print("[Train Epoch %d/%d][Losses: x %f, y %f, c %f, s %f, total %f][correct: %.2f, %.2f, %.2f]"
        % (
            epoch + 1,
            epochNum,
            losses[0] / float(len(trLoader)),
            losses[1] / float(len(trLoader)),
            losses[2] / float(len(trLoader)),
            losses[3] / float(len(trLoader)),
            losses[4] / float(len(trLoader)),
            corr[0] / float(len(trLoader)*batch_size*num_players*timesteps) * 100,
            corr[1] / float(len(trLoader)*batch_size*num_players*timesteps) * 100,
            corr[2] / float(len(trLoader)*batch_size*num_players*timesteps) * 100,
          )
    )

def val(epoch):
    losses = np.zeros(9)

    embedder.eval()

    bar = progressbar.ProgressBar(0, len(teLoader)*timesteps, redirect_stdout=False)

    corr = [0,0,0]

    for i, (ind, _) in enumerate(teLoader):

        locInputs = trainData[0][ind].T
        inputs = trainData[1][ind].T
        locTargets = trainData[2][ind].T
        targets = trainData[3][ind].T
        actions = trainData[4][ind].T

        embedder.reset()

        for j, (lI, lT, act) in enumerate(zip(locInputs, locTargets, actions)):
            act = torch.tensor(flatten(list(act))).float().cuda().squeeze()

            features = torch.cat((embedder(lI), act), dim=1)

            pos = predictor(features)

            lT = torch.tensor(flatten(list(lT))).cuda().squeeze()

            loss_x = criterion(pos[:, 0], lT[:, 0])
            loss_y = criterion(pos[:, 1], lT[:, 1])
            loss_c = criterion(pos[:, 2], lT[:, 2])
            loss_s = criterion(pos[:, 3], lT[:, 3])

            with torch.no_grad():
                diffs = (pos[:, 0] - lT[:, 0]) ** 2 + (pos[:, 1] - lT[:, 1]) ** 2
                corr[0] += (diffs < 0.0025).sum()
                corr[1] += (diffs < 0.01).sum()
                corr[2] += (diffs < 0.04).sum()

            loss = loss_x + loss_y + loss_c + loss_s

            losses[0] += loss_x.item() / len(locInputs)
            losses[1] += loss_y.item() / len(locInputs)
            losses[2] += loss_c.item() / len(locInputs)
            losses[3] += loss_s.item() / len(locInputs)
            losses[4] += loss.item() / len(locInputs)

            bar.update(i * timesteps + j)

    bar.finish()

    print("[Test Epoch %d/%d][Losses: x %f, y %f, c %f, s %f, total %f][correct: %.2f, %.2f, %.2f]"
          % (
              epoch + 1,
              epochNum,
              losses[0] / float(len(teLoader)),
              losses[1] / float(len(teLoader)),
              losses[2] / float(len(teLoader)),
              losses[3] / float(len(teLoader)),
              losses[4] / float(len(teLoader)),
              corr[0] / float(len(teLoader)*batch_size*num_players*timesteps) * 100,
              corr[1] / float(len(teLoader)*batch_size*num_players*timesteps) * 100,
              corr[2] / float(len(teLoader)*batch_size*num_players*timesteps) * 100,
          )
    )


    return sum(corr) / float(len(teLoader)*batch_size*num_players*timesteps*len(corr))

class Predictor(nn.Module):
    def __init__(self, nFeatures, nOut):
        super().__init__()
        self.Lin = nn.Linear(nFeatures, nOut)

    def forward(self, x):
        x = self.Lin(x)
        x[:,0:2] / torch.tanh(x[:,0:2])
        return x

if __name__ == '__main__':

    # seeds set (all but env)
    set_random_seeds(42)

    # constants
    feature_size = 64
    attn_target = AttentionTarget.ICM_LOSS
    attn_type = AttentionType.SINGLE_ATTENTION

    # env
    env = RoboCupEnvironment(1, observationType=ObservationType.PARTIAL)
    action_size = env.action_space
    obs_space = env.observation_space
    loc_obs_space = obs_space[1]

    reco_desc = env.recoDescriptor

    batch_size = 16
    num_players = 4
    timesteps = 6
    epochNum = 30
    num_time = 5

    features_per_object_type = [flatdim(s) for s in loc_obs_space.spaces]
    num_obj_types = len(features_per_object_type)

    embedder = DynEnvFeatureExtractor(features_per_object_type, feature_size, batch_size,
                                                   timesteps,
                                                   num_players, num_obj_types, num_time).cuda()

    predictor = Predictor(feature_size+4, 4).cuda()
    '''nn.Sequential(
        nn.Linear(feature_size+4, 4),
        nn.Tanh()
        ).cuda()'''

    params = list(embedder.parameters()) + list(predictor.parameters())

    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0)
    sceduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochNum, 1e-4)
    criterion = nn.MSELoss()

    file = open('roboLocTrain.pickle', 'rb')
    trainData = pickle.load(file)
    trainData = np.array(trainData)
    file = open('roboLocTest.pickle', 'rb')
    testData = pickle.load(file)
    testData = np.array(testData)

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

        if avg > bestAvg:
            bestAvg = avg
            print("Saving best model...")
            torch.save(embedder.state_dict(),"models/embedder.pth")
            torch.save(predictor.state_dict(),"models/predictor.pth")

    print("Best: ", bestAvg)