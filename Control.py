import pymunk
import pygame
from pygame.locals import *
import pymunk.pygame_util
import sys
import time
from Ball import *
from Goalpost import *
from Robot import *

# Env type and settings
# Setup field, ball and robots
# Implement step function
# Render robot visions and truths

class Environment(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((900, 600))
        pygame.display.set_caption("Joints. Just wait and the L will tip over")
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        self.robots = [Robot(400,300,1),Robot(600,200,0)]
        self.ball = Ball(450,300)
        self.space.add(self.ball.shape.body,self.ball.shape)
        self.goalposts = [
            Goalpost(60,200),
            Goalpost(60,400),
            Goalpost(840,200),
            Goalpost(840,400),
        ]
        for goal in self.goalposts:
            self.space.add(goal.shape.body,goal.shape)
        for robot in self.robots:
            self.space.add(robot.shape.body,robot.shape)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def step(self,action):
        self.screen.fill((0,255,0))
        pygame.draw.line(self.screen,(255,255,255),(60,0),(60,600),2)
        pygame.draw.line(self.screen,(255,255,255),(840,0),(840,600),2)
        pygame.draw.line(self.screen,(255,255,255),(0,60),(900,60),2)
        pygame.draw.line(self.screen,(255,255,255),(0,540),(900,540),2)
        pygame.draw.line(self.screen,(255,255,255),(450,0),(450,600),2)
        pygame.draw.line(self.screen,(255,255,255),(60,150),(160,150),2)
        pygame.draw.line(self.screen,(255,255,255),(60,450),(160,450),2)
        pygame.draw.line(self.screen,(255,255,255),(160,150),(160,450),2)
        pygame.draw.line(self.screen,(255,255,255),(740,150),(840,150),2)
        pygame.draw.line(self.screen,(255,255,255),(740,450),(840,450),2)
        pygame.draw.line(self.screen,(255,255,255),(740,150),(740,450),2)
        pygame.draw.circle(self.screen,(255,255,255),(450,300),150,2)
        pygame.draw.circle(self.screen,(255,255,255),(450,300),5,0)
        pygame.draw.circle(self.screen,(255,255,255),(200,300),5,0)
        pygame.draw.circle(self.screen,(255,255,255),(700,300),5,0)

        self.space.debug_draw(self.draw_options)

        if action > 0:
            if action < 5:
                self.robots[0].step(action-1)
            elif action < 7:
                self.robots[0].turn(action-5)
            else:
                self.robots[0].kick(action-7)

        finished, reward = self.ball.isOutOfField()


        [robot.tick(1000/100.0) for robot in self.robots]
        [robot.isLeavingField() for robot in self.robots]

        self.space.step(1 / 100.0)

        pygame.display.flip()
        self.clock.tick(100)
        return reward, finished

    def ballMoved(self,pos):
        return pos

    def robotMoved(self,pos):
        return pos

    def getRobotVision(self,robot):
        return robot
