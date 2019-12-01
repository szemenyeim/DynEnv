from DynEnv import *
import pygame
from pygame.locals import *
import sys
import numpy as np
import random
import argparse
import cv2

# Launch game, allow user controls

def doRoboCup(args):
    env = RoboCupEnvironment(nPlayers=args.num_players, render=args.render, observationType=args.observationType,
                             noiseType=args.noiseType, noiseMagnitude=args.noiseMagnitude,
                             allowHeadTurn=args.use_continuous_actions)
    env.setRandomSeed(42)
    env.reset()
    env.renderMode = 'human'
    env.agentVisID = 1

    action0 = [0, 0, 0]
    action1 = [0, 0, 0]
    action2 = [0, 0, 0]

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.key == K_RETURN:
                    env.reset()
                elif event.key == K_w:
                    action1[0] = 1
                elif event.key == K_s:
                    action1[0] = 2
                elif event.key == K_d:
                    action1[0] = 3
                elif event.key == K_a:
                    action1[0] = 4
                elif event.key == K_q:
                    action1[1] = 1
                elif event.key == K_e:
                    action1[1] = 2
                elif event.key == K_r:
                    action1[2] = 1
                elif event.key == K_f:
                    action1[2] = 2
                elif event.key == K_UP:
                    action2[0] = 1
                elif event.key == K_DOWN:
                    action2[0] = 2
                elif event.key == K_LEFT:
                    action2[0] = 3
                elif event.key == K_RIGHT:
                    action2[0] = 4
                elif event.key == K_DELETE:
                    action2[1] = 1
                elif event.key == K_PAGEDOWN:
                    action2[1] = 2
                elif event.key == K_END:
                    action2[2] = 1
                elif event.key == K_HOME:
                    action2[2] = 2
                elif event.key == K_RETURN:
                    env.reset()
            elif event.type == KEYUP:
                action1 = [0, 0, 0]
                action2 = [0, 0, 0]

        action = np.array([action1, ] + [action0,] * (args.num_players-1) + [action2,] + [action0,] * (args.num_players-1))
        #a1 = np.stack((np.random.randint(0,5,(nPlayers*2)),np.random.randint(0,3,(nPlayers*2)),np.random.randint(0,3,(nPlayers*2)))).T
        ret = env.step(action)
        rend = env.render()
        if ret[2]:
            break

def doDrive(args):
    env = DrivingEnvironment(nPlayers=args.num_players, render=args.render, observationType=args.observationType,
                             noiseType=args.noiseType, noiseMagnitude=args.noiseMagnitude,
                             continuousActions=args.use_continuous_actions)
    env.setRandomSeed(42)
    env.reset()
    env.renderMode = 'human'
    env.agentVisID = 0

    #action1 = [random.randint(0,2), random.randint(0,2)]
    action1 = [1,1]
    action = np.array([action1,]*(args.num_players))

    gameCnt = 0
    obsCnt = 0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.key == K_RETURN:
                    env.reset()
                elif event.key == K_w:
                    action[(0,0)] = 2
                elif event.key == K_s:
                    action[(0,0)] = 0
                elif event.key == K_d:
                    action[(0,1)] = 0
                elif event.key == K_a:
                    action[(0,1)] = 2

            elif event.type == KEYUP:
                action[(0,0)] = 1
                action[(0,1)] = 1

        #a1 = np.random.randint(0,3,(nPlayers*2,2))
        ret = env.step(action)
        rend = env.render()

        '''for img in rend[0]:
            cv2.imwrite("../../images/drive/game/img%d.png" % gameCnt, img)
            gameCnt += 1
        for img in rend[1]:
            cv2.imwrite("../../images/drive/obs/img%d.png" % obsCnt, img)
            obsCnt += 1'''

        if ret[2]:
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Play with the env')

    # env Params
    parser.add_argument('--env', default=DynEnvType.DRIVE, type=DynEnvType.argparse, choices=list(DynEnvType),
                        help='Environment type')
    parser.add_argument('--num-players', type=int, default=2, metavar='NUM_PLAYERS',
                        help='number of players in the environment [1-5]')
    parser.add_argument('--observationType', default=ObservationType.PARTIAL, type=ObservationType.argparse, choices=list(ObservationType),
                        help='Observation type')
    parser.add_argument('--noiseType', default=NoiseType.REALISTIC, type=NoiseType.argparse, choices=list(NoiseType),
                        help='Noise type')
    parser.add_argument('--noiseMagnitude', type=float, default=1.0,
                        help='Noise magnitude [0-5]')
    parser.add_argument('--use-continuous-actions', type=bool, default=False, metavar='CONTINUOUS',
                        help='Enable continuous actions (all actions in driving, head turning in robot soccer)')
    parser.add_argument('--render', type=bool, default=True, metavar='RENDER',
                        help='Enable rendering')

    args = parser.parse_args()

    drive = args.env

    pygame.init()

    if drive == DynEnvType.DRIVE:
        doDrive(args)
    else:
        doRoboCup(args)