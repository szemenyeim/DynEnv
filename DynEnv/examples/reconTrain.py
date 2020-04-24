import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunkoptions

pymunkoptions.options["debug"] = False
from DynEnv import *
from DynEnv.models.agent import ICMAgent
from DynEnv.models import DynEnvFeatureExtractor, ReconNet
from DynEnv.utils.utils import set_random_seeds, AttentionTarget, AttentionType, flatten
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import progressbar
from gym.spaces import flatdim
import torch.nn as nn

def train(epoch):

    losses = np.zeros(5)
    reconLosses = np.zeros(9)

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
        objEmbedder.reset()

        rec_optimizer.zero_grad()

        for j, (lI,lT,act,I) in enumerate(zip(locInputs, locTargets, actions, inputs)):

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

            if not localization:
                obj_features = objEmbedder(I, pos.detach())

            bar.update(i*timesteps + j)

        if not localization:
            recLosses = reconstructor(obj_features, targets[-1])
            recLosses.loss.backward()
            rec_optimizer.step()

            reconLosses[0] += recLosses.x
            reconLosses[1] += recLosses.y
            reconLosses[2] += recLosses.confidence
            reconLosses[3] += recLosses.binary
            reconLosses[4] += recLosses.continuous
            reconLosses[5] += recLosses.cls
            reconLosses[6] += recLosses.loss.item()
            reconLosses[7] += recLosses.recall
            reconLosses[8] += recLosses.precision

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

    print("[Reconstrction] "f"Recon Loss: {reconLosses[6]/float(len(trLoader)):.4f}, X: {reconLosses[0]/float(len(trLoader)):.4f},"
          f" Y: {reconLosses[1]/float(len(trLoader)):.4f}, " f"Conf: {reconLosses[2]/float(len(trLoader)):.4f}," 
          f" Bin: {reconLosses[3]/float(len(trLoader)):.4f}, Cont: {reconLosses[4]/float(len(trLoader)):.4f}, "
          f"Cls: {reconLosses[5]/float(len(trLoader)):.4f} " f"  [Recall: {reconLosses[7]/float(len(trLoader)) * 100.0:.2f}, "
          f"Precision: {reconLosses[8]/float(len(trLoader)) * 100.0:.2f}]")

def val(epoch):

    losses = np.zeros(5)
    reconLosses = np.zeros(9)

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

        for j, (lI, lT, act, I) in enumerate(zip(locInputs, locTargets, actions, inputs)):
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

            if not localization:
                obj_features = objEmbedder(I, pos.detach())

            bar.update(i * timesteps + j)

        if not localization:
            recLosses = reconstructor(obj_features, targets[-1])

            reconLosses[0] += recLosses.x
            reconLosses[1] += recLosses.y
            reconLosses[2] += recLosses.confidence
            reconLosses[3] += recLosses.binary
            reconLosses[4] += recLosses.continuous
            reconLosses[5] += recLosses.cls
            reconLosses[6] += recLosses.loss.item()
            reconLosses[7] += recLosses.recall
            reconLosses[8] += recLosses.precision

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

    print("[Reconstrction] "f"Recon Loss: {reconLosses[6]/float(len(teLoader)):.4f}, X: {reconLosses[0]/float(len(teLoader)):.4f},"
          f" Y: {reconLosses[1]/float(len(teLoader)):.4f}, " f"Conf: {reconLosses[2]/float(len(teLoader)):.4f}," 
          f" Bin: {reconLosses[3]/float(len(teLoader)):.4f}, Cont: {reconLosses[4]/float(len(teLoader)):.4f}, "
          f"Cls: {reconLosses[5]/float(len(teLoader)):.4f} " f"  [Recall: {reconLosses[7]/float(len(teLoader)) * 100.0:.2f}, "
          f"Precision: {reconLosses[8]/float(len(teLoader)) * 100.0:.2f}]")

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

    localization = False
    baseName = "roboLoc" if localization else "roboRec"

    # constants
    feature_size = 64
    attn_target = AttentionTarget.ICM_LOSS
    attn_type = AttentionType.SINGLE_ATTENTION

    # env
    env = RoboCupEnvironment(1, observationType=ObservationType.PARTIAL)
    action_size = env.action_space
    obs_space = env.observation_space
    loc_obs_space = obs_space[1]
    obj_obs_space = obs_space[0]

    reco_desc = env.recoDescriptor

    batch_size = 16 if localization else 8
    num_players = 4
    timesteps = 6 if localization else 9
    epochNum = 30
    num_time = 5

    features_per_object_type = [flatdim(s) for s in loc_obs_space.spaces]
    num_obj_types = len(features_per_object_type)

    embedder = DynEnvFeatureExtractor(features_per_object_type, feature_size, batch_size,
                                                   timesteps,
                                                   num_players, num_obj_types, num_time).cuda()

    predictor = Predictor(feature_size+4, 4).cuda()

    if not localization:
        suffix = "Loc.pth"
        embedder.load_state_dict(torch.load("models/embedder" + suffix))
        predictor.load_state_dict(torch.load("models/predictor" + suffix))

    features_per_object_type = [flatdim(s) for s in obj_obs_space.spaces]
    num_obj_types = len(features_per_object_type)
    objEmbedder = DynEnvFeatureExtractor(features_per_object_type, feature_size, batch_size,
                                                   timesteps,
                                                   num_players, num_obj_types, num_time, extended_feature_cnt=4).cuda()
    reconstructor = ReconNet(feature_size, reco_desc).cuda()

    params = list(embedder.parameters()) + list(predictor.parameters())

    loc_lr = 1e-3 if localization else 1e-4
    rec_lr = 1e-3

    optimizer = torch.optim.Adam(params, lr=loc_lr, weight_decay=0)
    sceduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochNum, loc_lr/10)
    criterion = nn.MSELoss()

    params = list(objEmbedder.parameters()) + list(reconstructor.parameters())

    rec_optimizer = torch.optim.Adam(params, lr=rec_lr, weight_decay=0)
    rec_sceduler = torch.optim.lr_scheduler.CosineAnnealingLR(rec_optimizer, epochNum, rec_lr/10)

    file = open(baseName + 'Train.pickle', 'rb')
    trainData = pickle.load(file)
    trainData = np.array(trainData)
    file = open(baseName + 'Test.pickle', 'rb')
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

        if not localization:
            rec_sceduler.step()

        if avg > bestAvg:
            bestAvg = avg
            print("Saving best model...")
            suffix = "Loc.pth" if localization else "Rec.pth"
            torch.save(embedder.state_dict(),"models/embedder" + suffix)
            torch.save(predictor.state_dict(),"models/predictor" + suffix)

    print("Best: ", bestAvg)