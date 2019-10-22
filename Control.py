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
    def __init__(self,nPlayers = 1,render=False):
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

        self.randBase = 0.05

        self.maxVisDist = self.W/2
        self.maxVisAngle = math.pi/12

        self.maxPlayers = 6
        self.nPlayers = min(nPlayers,self.maxPlayers)
        self.render = render
        self.timeStep = 100.0

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        self.lines = [
            # Outer lines
            (pymunk.Vec2d(self.sideLength,          self.sideLength),pymunk.Vec2d(self.sideLength,          self.H - self.sideLength)),
            (pymunk.Vec2d(self.W - self.sideLength, self.sideLength),pymunk.Vec2d(self.W - self.sideLength, self.H - self.sideLength)),
            (pymunk.Vec2d(self.sideLength,          self.sideLength),pymunk.Vec2d(self.W - self.sideLength, self.sideLength)),
            (pymunk.Vec2d(self.sideLength, self.H - self.sideLength),pymunk.Vec2d(self.W - self.sideLength, self.H - self.sideLength)),
            # Middle Line
            (pymunk.Vec2d(self.W / 2, self.sideLength),pymunk.Vec2d(self.W / 2, self.H - self.sideLength)),
            # Penalty Box #1
            (pymunk.Vec2d(self.sideLength,                      self.H / 2 - self.penaltyWidth), pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 - self.penaltyWidth)),
            (pymunk.Vec2d(self.sideLength,                      self.H / 2 + self.penaltyWidth), pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 + self.penaltyWidth)),
            (pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 - self.penaltyWidth), pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 + self.penaltyWidth)),
            # Penalty Box #2
            (pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 - self.penaltyWidth), pymunk.Vec2d(self.W - self.sideLength,                      self.H / 2 - self.penaltyWidth)),
            (pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 + self.penaltyWidth), pymunk.Vec2d(self.W - self.sideLength,                      self.H / 2 + self.penaltyWidth)),
            (pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 - self.penaltyWidth), pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 + self.penaltyWidth))
        ]

        self.centerCircle = [pymunk.Vec2d((self.W // 2, self.H // 2)), self.centerCircleRadius]

        self.fieldCrosses = [
            (pymunk.Vec2d(self.W // 2,                                   self.H // 2), self.penaltyRadius),
            (pymunk.Vec2d(self.sideLength + self.penaltyDist,            self.H // 2), self.penaltyRadius),
            (pymunk.Vec2d(self.W - (self.sideLength + self.penaltyDist), self.H // 2), self.penaltyRadius),
        ]

        centX = self.W/2

        self.robotSpots = [
            (centX-(self.ballRadius*2+Robot.totalRadius),self.H/2+10),
            (centX-(Robot.totalRadius+self.lineWidth/2+self.centerCircleRadius),self.sideLength + self.fieldH/4),
            (centX-(Robot.totalRadius+self.lineWidth/2+self.centerCircleRadius),self.sideLength + 3*self.fieldH/4),
            (centX-(self.sideLength + self.fieldW/4),self.sideLength + self.fieldH/4),
            (centX-(self.sideLength + self.fieldW/4),self.sideLength + 3*self.fieldH/4),
            ((self.sideLength), self.H/2),
            (centX+(self.centerCircleRadius*2+Robot.totalRadius+self.lineWidth/2),self.H/2),
            (centX+(Robot.totalRadius+self.lineWidth/2+self.centerCircleRadius),self.sideLength + self.fieldH/4),
            (centX+(Robot.totalRadius+self.lineWidth/2+self.centerCircleRadius),self.sideLength + 3*self.fieldH/4),
            (centX+(self.sideLength + self.fieldW/4),self.sideLength + self.fieldH/4),
            (centX+(self.sideLength + self.fieldW/4),self.sideLength + 3*self.fieldH/4),
            (self.W - (self.sideLength), self.H/2),
        ]

        self.robots = [Robot(spot,0) for i,spot in enumerate(self.robotSpots[:self.maxPlayers]) if i < self.nPlayers] \
                      + [Robot(spot,1) for i,spot in enumerate(self.robotSpots[self.maxPlayers:]) if i < self.nPlayers]
        for robot in self.robots:
            self.space.add(robot.leftFoot.body,robot.leftFoot,robot.rightFoot.body,robot.rightFoot,robot.joint,robot.rotJoint)

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

        if self.render:
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Robot Soccer")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)


    def robotPushingDet(self,arbiter, space, data):
        robot1 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        robot2 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[1] or robot.rightFoot == arbiter.shapes[1]))

        if not robot1.touching or not robot2.touching:
            print("Robot Collision")

            v1 = arbiter.shapes[0].body.velocity
            v2 = arbiter.shapes[1].body.velocity
            p1 = robot1.getPos()
            p2 = robot2.getPos()
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
            robot1.touching = True
            robot2.touching = True
            robot1.touchCntr = 0
            robot2.touchCntr = 0

        robot1.touchCntr += 1
        robot2.touchCntr += 1
        normalThresh = 0.99995 ** robot1.touchCntr
        pushingThresh = 0.99998 ** robot1.touchCntr
        r = random.random()
        if r > (pushingThresh if robot1.mightPush else normalThresh) and not robot1.fallen:
            robot1.fall(self)
            robot1.touchCntr = 0
        r = random.random()
        if r > (pushingThresh if robot2.mightPush else normalThresh) and not robot2.fallen:
            robot2.fall(self)
            robot2.touchCntr = 0

        if robot1.mightPush and not robot2.mightPush and robot2.fallen:
            print("Robot 1 Pushing")
            robot1.penalize(5000,self)
            robot1.touchCntr = 0
        elif robot2.mightPush and not robot1.mightPush and robot1.fallen:
            print("Robot 2 Pushing")
            robot2.penalize(5000,self)
            robot2.touchCntr = 0


    def separate(self,arbiter, space, data):
        print("Collision Separated")
        robots = [robot for robot in self.robots if (robot.leftFoot in arbiter.shapes or robot.rightFoot in arbiter.shapes)]
        for robot in robots:
            robot.touching = False
            robot.mightPush = False
            robot.touchCntr = 0


    def goalpostCollision(self,arbiter, space, data):
        robot = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        if robot.fallen:
            robot.touchCntr = 0
            return
        if not robot.touching:
            print("Goalpost Collision")
            robot.touching = True
            robot.touchCntr = 0
        robot.touchCntr += 1
        pushingThresh = 0.9999 ** robot.touchCntr
        r = random.random()
        if r > pushingThresh:
            robot.fall(self)


    def ballCollision(self,arbiter, space, data):
        robot = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        team = robot.team
        #print("Ball Kick", team)
        self.ball.lastKicked = team
        return True

    def drawStaticObjects(self):
        self.screen.fill((0, 255, 0))

        for line in self.lines:
            pygame.draw.line(self.screen,(255,255,255),line[0],line[1],self.lineWidth)

        for cross in self.fieldCrosses:
            pygame.draw.circle(self.screen,(255,255,255),cross[0],cross[1]*2,0)

        pygame.draw.circle(self.screen, (255,255,255), self.centerCircle[0], self.centerCircle[1] * 2, self.lineWidth)

        self.space.debug_draw(self.draw_options)


    def step(self,actions):
        t1 = time.clock()

        for i in range(50):
            if self.render:
                self.drawStaticObjects()

            if len(actions) != len(self.robots):
                print("Error: There must be action s for every robot")

            for action, robot in zip(actions,self.robots):
                if i == 0:
                    self.processAction(action,robot)
                robot.tick(1000 / self.timeStep, self.ball.shape.body.position, self.space, self)
                robot.isLeavingField(self)

            finished, reward = self.ball.isOutOfField(self)

            self.space.step(1 / self.timeStep)

            if i % 10 == 9:
                for robot in self.robots:
                    self.getRobotVision(robot)

            if self.render:
                pygame.display.flip()
                self.clock.tick(self.timeStep)
        t2 = time.clock()
        print((t2-t1)*1000)
        return reward, finished


    def processAction(self, action, robot):
        if action > 0:
            if action < 5:
                robot.step(action-1,self)
            elif action < 7:
                robot.turn(action-5,self)
            elif action >= 7:
                robot.kick(action-7,self)

    def getRobotVision(self,robot):
        pos = robot.getPos()
        angle = robot.leftFoot.body.angle
        headAngle = robot.headAngle

        angle1 = angle+headAngle+robot.fieldOfView
        vec1 = pymunk.Vec2d(1,0)
        vec1.rotate(angle1)

        angle2 = angle+headAngle-robot.fieldOfView
        vec2 = pymunk.Vec2d(1,0)
        vec2.rotate(angle2)

        ballPos = self.ball.shape.body.position - pos

        ball, ballDet = isSeenInArea(ballPos,vec1,vec2,self.maxVisDist,self.ballRadius)

        maxDistSqr = self.maxVisDist**2

        robDets = [isSeenInArea(rob.getPos() - pos,vec1,vec2,self.maxVisDist,Robot.totalRadius) for rob in self.robots if robot != rob]
        goalDets = [isSeenInArea(goal.shape.body.position - pos,vec1,vec2,self.maxVisDist,self.goalPostRadius) for goal in self.goalposts]
        crossDets = [isSeenInArea(cross[0] - pos,vec1,vec2,self.maxVisDist,self.penaltyRadius) for cross in self.fieldCrosses]
        lineDets = [isLineInArea(p1 - pos,p2 - pos,vec1,vec2,self.maxVisDist,maxDistSqr) for p1,p2 in self.lines]

        circlePos = self.centerCircle[0] - pos
        circleDets = (isSeenInArea(circlePos,vec1,vec2,self.maxVisDist,self.centerCircleRadius),self.centerCircleRadius)


        robRobInter = [[doesInteract(rob1,rob2,Robot.totalRadius) for _,rob1 in robDets] for _,rob2 in robDets]
        robBallInter = [doesInteract(rob,ballDet,Robot.totalRadius) for _,rob in robDets]
        robPostInter = [[doesInteract(rob,post,Robot.totalRadius) for _,rob in robDets] for _,post in goalDets]
        robCrossInter = [[doesInteract(rob,cross,Robot.totalRadius) for _,rob in robDets] for _,cross in crossDets]
        ballPostInter = [doesInteract(ballDet,post,self.ballRadius,False) for _,post in goalDets]
        ballCrossInter = [doesInteract(ballDet,cross,self.ballRadius,False)for _,cross in crossDets]

        # Remove occlusion
        ballDets = [(ball,ballDet,self.ballRadius)] if ball != 0 and 2 not in robBallInter else []
        robDets = [(dtype,robDet,Robot.totalRadius) for i,(dtype,robDet) in enumerate(robDets) if dtype != 0 and 2 not in robRobInter[i]]
        crossDets = [(dtype,crossDet,self.penaltyRadius) for i,(dtype,crossDet)in enumerate(crossDets) if dtype != 0 and 2 not in robCrossInter[i]]
        goalDets = [(dtype,postDet,self.goalPostRadius) for i,(dtype,postDet) in enumerate(goalDets) if dtype != 0 and 2 not in robPostInter[i]]
        lineDets = [(dtype,pt1,pt2) for dtype,(pt1,pt2) in lineDets if dtype != 0]

        noiseType = 2
        if noiseType > 0:

            rand = self.randBase if noiseType == 1 else self.randBase/2

            # Random position noise and false negatives
            ballDets = [addNoise(ball, noiseType) for ball in ballDets if random.random() > rand]
            robDets = [addNoise(robot, noiseType) for robot in robDets if random.random() > rand]
            crossDets = [addNoise(cross, noiseType) for cross in crossDets if random.random() > rand]
            goalDets = [addNoise(goal, noiseType) for goal in goalDets if random.random() > rand]
            lineDets = [addNoiseLine(line, noiseType) for line in lineDets if random.random() > rand]

            # Random false positives
            for i in range(10):
                if random.random() < rand:
                    c = random.randint(0,5)
                    if c == 0:
                        ballDets.insert(random.randint(0,len(ballDets)),
                                       (3,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),self.ballRadius*2*random.random()))
                    elif c == 1:
                        robDets.insert(random.randint(0,len(robDets)),
                                       (3,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),Robot.totalRadius*2*random.random()))
                    elif c == 2:
                        crossDets.insert(random.randint(0,len(crossDets)),
                                       (3,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),self.penaltyRadius*2*random.random()))
                    elif c == 3:
                        goalDets.insert(random.randint(0,len(goalDets)),
                                       (3,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),self.goalPostRadius*2*random.random()))
                    elif c == 4:
                        lineDets.insert(random.randint(0,len(lineDets)),
                                       (3,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),
                                        pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5))))

        return ballDet,robDets,goalDets,crossDets,lineDets,circleDets
