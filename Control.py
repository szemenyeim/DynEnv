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
            self.space.add(robot.leftFoot.body,robot.leftFoot,robot.rightFoot.body,robot.rightFoot,robot.spring,robot.rotSpring)

        h = self.space.add_collision_handler(
            collision_types["robot"],
            collision_types["goalpost"])
        h.post_solve = self.goalpostCollision
        h.separate = self.separate
        h = self.space.add_collision_handler(
            collision_types["robot"],
            collision_types["robot"])
        h.pre_solve = self.robotPushingDet
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
            print("Robot Pushing detection")

            v1 = arbiter.shapes[0].body.velocity
            v2 = arbiter.shapes[1].body.velocity
            p1 = (robot1.leftFoot.body.position + robot1.rightFoot.body.position)/2.0
            p2 = (robot2.leftFoot.body.position + robot2.rightFoot.body.position)/2.0
            dp = p1-p2

            robot1.mightPush =  v1.length > 1 and math.cos(dp.angle-v1.angle) < -0.4
            robot2.mightPush =  v2.length > 1 and math.cos(dp.angle-v2.angle) > 0.4
        return True

    def robotCollision(self,arbiter, space, data):
        robot1 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        robot2 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[1] or robot.rightFoot == arbiter.shapes[1]))

        if not robot1.touching or not robot2.touching:
            print("Robot Collision")

            robot1.fall(self.space)
            robot2.fall(self.space)

            if robot1.mightPush and not robot2.mightPush:
                print("Robot 1 Pushing")
                robot1.penalize(5000)
            elif robot2.mightPush and not robot1.mightPush:
                print("Robot 2 Pushing")
                robot2.penalize(5000)

            robot1.mishtPush = False
            robot2.mishtPush = False

    def goalpostCollision(self,arbiter, space, data):
        robot = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        if not robot.touching:
            print("Goalpost Collision")
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

    def step(self,actions):
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

        if len(actions) != len(self.robots):
            print("Error: There must be action s for every robot")

        for action, robot in zip(actions,self.robots):
            self.processAction(action,robot)

        finished, reward = self.ball.isOutOfField()

        [robot.tick(1000/100.0, self.ball.shape.body.position,self.space) for robot in self.robots]
        [robot.isLeavingField() for robot in self.robots]

        self.space.step(1 / 100.0)

        pygame.display.flip()
        self.clock.tick(100)
        return reward, finished

    def processAction(self, action, robot):
        if action > 0:
            if action < 5:
                robot.step(action-1)
            elif action < 7:
                robot.turn(action-5)
            elif action == 7:
                robot.kick()

    def getRobotVision(self,robot):
        return robot
