import RoboEnv
import pygame
from pygame.locals import *
import sys

# Launch game, allow user controls

if __name__ == '__main__':
    nPlayers = 5
    env = RoboEnv.RoboEnv(nPlayers=nPlayers,render=False,observationType=RoboEnv.ObservationType.Partial,noiseType=RoboEnv.NoiseType.Realistic,noiseMagnitude = 2)
    pygame.init()
    action1 = [0,0,0,0]
    action2 = [0,0,0,0]
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
                elif event.key == K_1:
                    action1[2] = -6
                elif event.key == K_2:
                    action1[2] = -4
                elif event.key == K_3:
                    action1[2] = -2
                elif event.key == K_4:
                    action1[2] = -1
                elif event.key == K_5:
                    action1[2] = 0
                elif event.key == K_6:
                    action1[2] = 1
                elif event.key == K_7:
                    action1[2] = 2
                elif event.key == K_8:
                    action1[2] = 4
                elif event.key == K_9:
                    action1[2] = 6
                elif event.key == K_r:
                    action1[3] = 1
                elif event.key == K_f:
                    action1[3] = 2
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
                    action2[3] = 1
                elif event.key == K_HOME:
                    action2[3] = 2
            elif event.type == KEYUP:
                action1 = [0,0,0,0]
                action2 = [0,0,0,0]

        action = [action1,action2]*nPlayers
        ret = env.step(action)
        if ret[4]:
            print("Goal: reward: ",ret[0])