import pygame
import pymunk.pygame_util
import time
from Ball import *
from Goalpost import *
from Robot import *

# Env type and settings
# Setup field, ball and robots
# Implement step function
# Render robot visions and truths

class Environment(object):
    def __init__(self,nPlayers,render=False,gameType = GameType.Full,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic):
        pygame.init()

        self.gameType = gameType
        self.observationType = observationType
        self.noiseType = noiseType

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

        self.kickDiscount = 0.5

        self.randBase = 0.05

        self.maxVisDist = self.W/2
        self.maxVisAngle = math.pi/12

        self.maxPlayers = 6
        self.nPlayers = min(nPlayers,self.maxPlayers)
        self.render = render
        self.timeStep = 100.0

        self.teamRewards = [0,0]
        self.RobotRewards = [0,0]*self.nPlayers

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

        self.robots = [Robot(spot,0,i) for i,spot in enumerate(self.robotSpots[:self.maxPlayers]) if i < self.nPlayers] \
                      + [Robot(spot,1,self.nPlayers+i) for i,spot in enumerate(self.robotSpots[self.maxPlayers:]) if i < self.nPlayers]
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
            CollisionType.Robot,
            CollisionType.Goalpost)
        h.post_solve = self.goalpostCollision
        h.separate = self.separate

        h = self.space.add_collision_handler(
            CollisionType.Robot,
            CollisionType.Robot)
        h.begin = self.robotPushingDet
        h.post_solve = self.robotCollision
        h.separate = self.separate

        h = self.space.add_collision_handler(
            CollisionType.Robot,
            CollisionType.Ball)
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
        #print("Ball Kick", team)
        self.ball.lastKicked = [robot.id] + self.ball.lastKicked
        if len(self.ball.lastKicked) > 4:
            self.ball.lastKicked = self.ball.lastKicked[:4]
        return True

    def drawStaticObjects(self):
        self.screen.fill((0, 255, 0))

        for line in self.lines:
            pygame.draw.line(self.screen,(255,255,255),line[0],line[1],self.lineWidth)

        for cross in self.fieldCrosses:
            pygame.draw.circle(self.screen,(255,255,255),cross[0],cross[1]*2,0)

        pygame.draw.circle(self.screen, (255,255,255), self.centerCircle[0], self.centerCircle[1] * 2, self.lineWidth)

        self.space.debug_draw(self.draw_options)


    def isBallOutOfField(self):
        finished = False
        pos = self.ball.shape.body.position

        currReward = [0,0]

        outMin = self.sideLength - self.ballRadius
        outMaxX = self.W - self.sideLength + self.ballRadius
        outMaxY = self.H -self.sideLength + self.ballRadius

        # If ball is out
        if pos.y < outMin or pos.x < outMin or pos.y > outMaxY or pos.x > outMaxX:
            x = self.W/2
            y = self.H/2

            # If out on the sides
            if pos.y < outMin or pos.y > outMaxY:
                x = pos.x + 50 if self.ball.lastKicked[0] else pos.x - 50
                if pos.y < outMin:
                    y = outMin + self.ballRadius
                else:
                    y = outMaxY - self.ballRadius
            # If out on the ends
            else:
                # If goal
                if pos.y < self.H/2 + self.goalWidth and pos.y > self.H/2 - self.goalWidth:
                    finished = True
                    if pos.x < outMin:
                        currReward[0] += -1000
                        currReward[1] += 1000
                    else:
                        currReward[0] += 1000
                        currReward[1] += -1000
                # If simply out
                else:
                    # Handle two ends differently
                    if pos.x < outMin:
                        # Kick out
                        if self.ball.lastKicked[0]:
                            x = self.sideLength + self.penaltyLength
                        # Corner
                        else:
                            x = self.sideLength
                            y = self.sideLength if pos.y < self.H/2 else self.H-self.sideLength
                    else:
                        # Kick out
                        if not self.ball.lastKicked[0]:
                            x = self.W - (self.sideLength + self.penaltyLength)
                        # Corner
                        else:
                            x = self.W - self.sideLength
                            y = self.sideLength if pos.y < self.H/2 else self.H-self.sideLength
            # Move ball to middle and stop it
            self.ball.shape.body.position = pymunk.Vec2d(x,y)
            self.ball.shape.body.velocity = pymunk.Vec2d(0.0,0.0)
            self.ball.shape.body.angular_velocity = 0.0

        # Add ball movement to the reward
        if not finished:
            currReward[0] += self.ball.shape.body.position.x - pos.x
            currReward[1] -= self.ball.shape.body.position.x - pos.x

        # Create discounted personal rewards for the robots involved
        for i, id in enumerate(self.ball.lastKicked):
            self.robotRewards[id] += currReward[0] * (self.kickDiscount ** i) if id < self.nPlayers else currReward[1] * (self.kickDiscount ** i)

        # Create personal rewards for nearby robots not touching the ball, but only negative rewards
        for robot in self.robots:
            if robot.id not in self.ball.lastKicked and (robot.getPos() - pos).length < 150:
                self.robotRewards[robot.id] += min(0, currReward[0] * self.kickDiscount if robot.id < self.nPlayers else
                currReward[1] * self.kickDiscount)

        # Update team rewards
        self.teamRewards[0] += currReward[0]
        self.teamRewards[1] += currReward[1]
        
        return finished,reward

    def step(self,actions):
        t1 = time.clock()

        self.teamRewards = [0,0]
        self.robotRewards = [0,0]*self.nPlayers
        observations = []
        finished = False

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

            finished, reward = self.isBallOutOfField()

            self.space.step(1 / self.timeStep)

            if i % 10 == 9:
                observations.append([self.getRobotVision(robot) for robot in self.robots])

            if self.render:
                pygame.display.flip()
                self.clock.tick(self.timeStep)
        t2 = time.clock()
        print((t2-t1)*1000)
        return observations,self.teamRewards,self.robotRewards,finished


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

        ballDets = [isSeenInArea(ballPos,vec1,vec2,self.maxVisDist,self.ballRadius)]

        maxDistSqr = self.maxVisDist**2

        robDets = [isSeenInArea(rob.getPos() - pos,vec1,vec2,self.maxVisDist,Robot.totalRadius)+(robot.team == rob.team,) for rob in self.robots if robot != rob]
        goalDets = [isSeenInArea(goal.shape.body.position - pos,vec1,vec2,self.maxVisDist,self.goalPostRadius) for goal in self.goalposts]
        crossDets = [isSeenInArea(cross[0] - pos,vec1,vec2,self.maxVisDist,self.penaltyRadius) for cross in self.fieldCrosses]
        lineDets = [isLineInArea(p1 - pos,p2 - pos,vec1,vec2,self.maxVisDist,maxDistSqr) for p1,p2 in self.lines]

        circlePos = self.centerCircle[0] - pos
        circleDets = isSeenInArea(circlePos,vec1,vec2,self.maxVisDist,self.centerCircleRadius)


        robRobInter = [max([doesInteract(rob1,rob2,Robot.totalRadius) for _,rob1,_,_ in robDets if rob1 != rob2]) for _,rob2,_,_ in robDets]
        robBallInter = max([doesInteract(rob,ballDets[0][1],Robot.totalRadius) for _,rob,_,_ in robDets])
        robPostInter = [max([doesInteract(rob,post,Robot.totalRadius) for _,rob,_,_ in robDets]) for _,post,_ in goalDets]
        robCrossInter = [max([doesInteract(rob,cross,Robot.totalRadius) for _,rob,_,_ in robDets]) for _,cross,_ in crossDets]
        ballPostInter = max([doesInteract(ballDets[0][1],post,self.ballRadius,False) for _,post,_ in goalDets])
        ballCrossInter = [doesInteract(ballDets[0][1],cross,self.ballRadius,False)for _,cross,_ in crossDets]

        rand = self.randBase if self.noiseType == 1 else self.randBase/2

        # Random position noise and false negatives
        ballDets = [addNoise(ball, self.noiseType, max(robBallInter,ballPostInter), rand, True) for ball in ballDets]
        robDets = [addNoise(robot, self.noiseType, robRobInter[i], rand) for i,robot in enumerate(robDets)]
        goalDets = [addNoise(goal, self.noiseType, robPostInter[i], rand) for i,goal in enumerate(goalDets)]
        crossDets = [addNoise(cross, self.noiseType, max(robCrossInter[i], ballCrossInter[i]), rand, True) for i,cross in enumerate(crossDets)]
        lineDets = [addNoiseLine(line, self.noiseType, rand) for i,line in enumerate(lineDets)]
        circleDets = addNoise(circleDets, self.noiseType, 0, rand)

        for ball in ballDets:
            if ball[0] == SightingType.Misclassified:
                crossDets.append((SightingType.Normal,ball[1],ball[2]))
        for cross in crossDets:
            if cross[0] == SightingType.Misclassified:
                ballDets.append((SightingType.Normal,cross[1],cross[2]))

        # Random false positives
        for i in range(10):
            if random.random() < rand:
                c = random.randint(0,5)
                if c == 0:
                    ballDets.insert(len(ballDets),
                                   (SightingType.Normal,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),self.ballRadius*2*random.random()))
                elif c == 1:
                    robDets.insert(len(robDets),
                                   (SightingType.Normal,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),Robot.totalRadius*2*random.random(),random.random() > 0.5))
                elif c == 2:
                    goalDets.insert(len(goalDets),
                                   (SightingType.Normal,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),self.goalPostRadius*2*random.random()))
                elif c == 3:
                    crossDets.insert(len(crossDets),
                                   (SightingType.Normal,pymunk.Vec2d(self.maxVisDist*random.random(),self.maxVisDist*(random.random()-0.5)),self.penaltyRadius*2*random.random()))

        if self.noiseType == 2:
            for robot in robDets:
                if robot[0] == SightingType.Normal and random.random() < rand and robot[1].length < 150:
                    ballDets.insert(len(ballDets),
                                   (SightingType.Normal,pymunk.Vec2d(robot[1].x-Robot.totalRadius/2,robot[1].y-Robot.totalRadius/2),self.ballRadius+2*random.random()))


        # Remove occlusion
        ballDets = [ball for i,ball in enumerate(ballDets) if ball[0] != SightingType.NoSighting or ball[0] != SightingType.Misclassified]
        robDets = [robot for i,robot in enumerate(robDets) if robot[0] != SightingType.NoSighting]
        goalDets = [goal for i,goal in enumerate(goalDets) if goal[0] != SightingType.NoSighting]
        crossDets = [cross for i,cross in enumerate(crossDets) if cross[0] != SightingType.NoSighting or cross[0] != SightingType.Misclassified]
        lineDets = [line for i,line in enumerate(lineDets) if line[0] != SightingType.NoSighting]

        return ballDets,robDets,goalDets,crossDets,lineDets,circleDets
