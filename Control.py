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

        self.W = 1040
        self.H = 740
        self.fieldW = 900
        self.fieldH = 600
        self.sideLength = 70
        self.lineWidth = 5
        self.penaltyRadius = 5
        self.penaltyLength = 60
        self.penaltyWidth = 110
        self.penaltyDist = 130
        self.centerCircleRadius = 75
        self.goalWidth = 80
        self.goalPostRadius = 5
        self.ballRadius = 5

        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Robot Soccer")
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        self.robots = [Robot(self.W/2-25,self.H/2-10,1),Robot(self.W/2+50,self.H/2,0)]
        self.ball = Ball(self.W/2,self.H/2,self.ballRadius)
        self.space.add(self.ball.shape.body,self.ball.shape)
        self.goalposts = [
            Goalpost(self.sideLength,self.H/2+self.goalWidth,self.goalPostRadius),
            Goalpost(self.sideLength,self.H/2-self.goalWidth,self.goalPostRadius),
            Goalpost(self.W-self.sideLength,self.H/2+self.goalWidth,self.goalPostRadius),
            Goalpost(self.W-self.sideLength,self.H/2-self.goalWidth,self.goalPostRadius),
        ]
        for goal in self.goalposts:
            self.space.add(goal.shape.body,goal.shape)
        for robot in self.robots:
            self.space.add(robot.leftFoot.body,robot.leftFoot,robot.rightFoot.body,robot.rightFoot,robot.joint,robot.rotJoint)

        h = self.space.add_collision_handler(
            collision_types["robot"],
            collision_types["goalpost"])
        h.post_solve = self.goalpostCollision
        h.separate = self.separate
        h = self.space.add_collision_handler(
            collision_types["robot"],
            collision_types["robot"])
        h.begin = self.robotPushingDet
        h.post_solve = self.robotCollision
        h.separate = self.separate
        h = self.space.add_collision_handler(
            collision_types["robot"],
            collision_types["ball"])
        h.begin = self.ballCollision

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def robotPushingDet(self,arbiter, space, data):
        robot1 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        robot2 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[1] or robot.rightFoot == arbiter.shapes[1]))

        if not robot1.touching or not robot2.touching:
            print("Robot Collision")

            v1 = arbiter.shapes[0].body.velocity
            v2 = arbiter.shapes[1].body.velocity
            p1 = (robot1.leftFoot.body.position + robot1.rightFoot.body.position)/2.0
            p2 = (robot2.leftFoot.body.position + robot2.rightFoot.body.position)/2.0
            dp = p1-p2

            robot1.mightPush = v1.length > 1 and math.cos(dp.angle-v1.angle) < -0.4
            robot2.mightPush = v2.length > 1 and math.cos(dp.angle-v2.angle) > 0.4
        return True

    def robotCollision(self,arbiter, space, data):
        robot1 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        robot2 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[1] or robot.rightFoot == arbiter.shapes[1]))

        if robot1 == robot2:
            return

        if not robot1.touching or not robot2.touching:

            normalThresh = 0.8
            pushingThresh = 0.9
            robot1.touching = True
            robot2.touching = True
            r = random.random()
            if r > (pushingThresh if robot1.mightPush else normalThresh):
                robot1.fall(self.space)
            r = random.random()
            if r > (pushingThresh if robot2.mightPush else normalThresh):
                robot2.fall(self.space)

            if robot1.mightPush and not robot2.mightPush and robot2.fallen:
                print("Robot 1 Pushing")
                robot1.penalize(5000,self)
            elif robot2.mightPush and not robot1.mightPush and robot1.fallen:
                print("Robot 2 Pushing")
                robot2.penalize(5000,self)

    def goalpostCollision(self,arbiter, space, data):
        robot = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        if not robot.touching:
            print("Goalpost Collision")
            robot.touching = True
            pushingThresh = 0.9
            r = random.random()
            if r > pushingThresh:
                robot.fall(self.space)

    def ballCollision(self,arbiter, space, data):
        robot = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        team = robot.team
        #print("Ball Kick", team)
        self.ball.lastKicked = team
        return True

    def separate(self,arbiter, space, data):
        print("Collision Separated")
        robots = [robot for robot in self.robots if (robot.leftFoot in arbiter.shapes or robot.rightFoot in arbiter.shapes)]
        for robot in robots:
            robot.touching = False
            robot.mightPush = False

    def step(self,actions):
        self.screen.fill((0,255,0))
        pygame.draw.line(self.screen,(255,255,255),(self.sideLength,self.sideLength),(self.sideLength,self.H-self.sideLength),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.W-self.sideLength,self.sideLength),(self.W-self.sideLength,self.H-self.sideLength),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.sideLength,self.sideLength),(self.W-self.sideLength,self.sideLength),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.sideLength,self.H-self.sideLength),(self.W-self.sideLength,self.H-self.sideLength),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.W/2,self.sideLength),(self.W/2,self.H-self.sideLength),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.sideLength,self.H/2-self.penaltyWidth),(self.sideLength+self.penaltyLength,self.H/2-self.penaltyWidth),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.sideLength,self.H/2+self.penaltyWidth),(self.sideLength+self.penaltyLength,self.H/2+self.penaltyWidth),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.sideLength+self.penaltyLength,self.H/2-self.penaltyWidth),(self.sideLength+self.penaltyLength,self.H/2+self.penaltyWidth),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.W-self.sideLength-self.penaltyLength,self.H/2-self.penaltyWidth),(self.W-self.sideLength,self.H/2-self.penaltyWidth),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.W-self.sideLength-self.penaltyLength,self.H/2+self.penaltyWidth),(self.W-self.sideLength,self.H/2+self.penaltyWidth),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.W-self.sideLength-self.penaltyLength,self.H/2-self.penaltyWidth),(self.W-self.sideLength-self.penaltyLength,self.H/2+self.penaltyWidth),self.lineWidth)
        pygame.draw.circle(self.screen,(255,255,255),(self.W//2,self.H//2),self.centerCircleRadius*2,self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),(self.W//2-self.penaltyRadius,self.H//2),(self.W//2+self.penaltyRadius,self.H//2),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),
                         (self.sideLength+self.penaltyDist-self.penaltyRadius,self.H//2),
                         (self.sideLength+self.penaltyDist+self.penaltyRadius,self.H//2),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),
                         (self.sideLength+self.penaltyDist,self.H//2-self.penaltyRadius),
                         (self.sideLength+self.penaltyDist,self.H//2+self.penaltyRadius),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),
                         (self.W-(self.sideLength+self.penaltyDist-self.penaltyRadius),self.H//2),
                         (self.W-(self.sideLength+self.penaltyDist+self.penaltyRadius),self.H//2),self.lineWidth)
        pygame.draw.line(self.screen,(255,255,255),
                         (self.W-(self.sideLength+self.penaltyDist),self.H//2-self.penaltyRadius),
                         (self.W-(self.sideLength+self.penaltyDist),self.H//2+self.penaltyRadius),self.lineWidth)

        self.space.debug_draw(self.draw_options)

        if len(actions) != len(self.robots):
            print("Error: There must be action s for every robot")

        for action, robot in zip(actions,self.robots):
            self.processAction(action,robot)

        finished, reward = self.ball.isOutOfField(self)

        [robot.tick(1000/100.0, self.ball.shape.body.position,self.space,self) for robot in self.robots]
        [robot.isLeavingField(self) for robot in self.robots]

        self.space.step(1 / 100.0)

        pygame.display.flip()
        self.clock.tick(100)
        return reward, finished

    def processAction(self, action, robot):
        if action > 0:
            if action < 5:
                robot.step(action-1,self.space)
            elif action < 7:
                robot.turn(action-5,self.space)
            elif action >= 7:
                robot.kick(action-7,self.space)

    def getRobotVision(self,robot):
        return robot
