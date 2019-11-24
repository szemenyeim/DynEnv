import DynEnv
import pygame
from pygame.locals import *
import sys
import numpy as np
import random

# Launch game, allow user controls

def doRoboCup():
    nPlayers = 1
    env = DynEnv.RoboCupEnvironment(nPlayers=nPlayers, render=True, observationType=DynEnv.ObservationType.Partial,
                                    noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=2)
    env.setRandomSeed(42)

    action1 = [0, 0, 0]
    action2 = [0, 0, 0]

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)
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

        action = np.array([action1, action2] * nPlayers)
        ret = env.step(action)

def doDrive():
    nPlayers = 1
    env = DynEnv.DrivingEnvironment(nPlayers=nPlayers, render=True, observationType=DynEnv.ObservationType.Partial,
                                    noiseType=DynEnv.NoiseType.Realistic, noiseMagnitude=2)
    env.setRandomSeed(42)
    env.reset()

    #action1 = [random.randint(0,2), random.randint(0,2)]
    action1 = [1,1]
    action = np.array([action1,]*(nPlayers*2))

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

        ret = env.step(action)
        if ret[2]:
            break

if __name__ == '__main__':

    drive = False

    pygame.init()

    if drive:
        doDrive()
    else:
        doRoboCup()