import pygame
import pymunk.pygame_util
import time
from .utils import *
from .cutils import *
from .Ball import Ball
from .Goalpost import Goalpost
from .Robot import Robot
import cv2
import numpy as np


class Environment(object):

    def __init__(self,nPlayers,render=False,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic, noiseMagnitude = 2):

        # Basic settings
        self.observationType = observationType
        self.noiseType = noiseType
        self.maxPlayers = 5
        self.nPlayers = min(nPlayers,self.maxPlayers)
        self.render = render

        # Which robot's observation to visualize
        self.visId = 0#random.randint(0,self.nPlayers*2-1)

        # Field setup
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

        # Vision settings
        if noiseMagnitude < 0 or noiseMagnitude > 5:
            print("Error: The noise magnitude must be between 0 and 5!")
            exit(0)
        if observationType == ObservationType.Full and noiseMagnitude > 0:
            print("Warning: Full observation type does not support noisy observations, but your noise magnitude is set to a non-zero value! (The noise setting has no effect in this case)")
        self.randBase = 0.01 * noiseMagnitude
        self.noiseMagnitude = noiseMagnitude
        self.maxVisDist = [self.W*0.4,self.W*0.8]

        # Free kick status
        self.ballOwned = 1
        self.ballFreeCntr = 9999
        self.gracePeriod = 0

        # Bookkeeping for robots inside penalty area for illegal defender
        self.defenders = [[],[]]

        # Reward settings
        self.kickDiscount = 0.5
        self.teamRewards = [0,0]
        self.RobotRewards = [0,0]*self.nPlayers
        self.penalTimes = [10000,10000]

        # Simulation settings
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.timeStep = 100.0

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
            # Kickoff team
            (centX-(self.ballRadius*2+Robot.totalRadius),self.H/2+(10 if random.random() > 0.5 else -10)),
            (centX-(Robot.totalRadius+self.lineWidth/2),self.sideLength + (self.fieldH/4-20 if random.random() > 0.5 else 3*self.fieldH/4+20)),
            (centX-(self.sideLength + self.fieldW/4),self.sideLength + self.fieldH/4),
            (centX-(self.sideLength + self.fieldW/4),self.sideLength + 3*self.fieldH/4),
            ((self.sideLength), self.H/2),
            # Opposing team
            (centX+(self.centerCircleRadius*2+Robot.totalRadius+self.lineWidth/2),self.H/2),
            (centX+(Robot.totalRadius+self.lineWidth/2+self.centerCircleRadius),self.sideLength + self.fieldH/4),
            (centX+(Robot.totalRadius+self.lineWidth/2+self.centerCircleRadius),self.sideLength + 3*self.fieldH/4),
            (centX+(self.sideLength + self.fieldW/4),self.sideLength + self.fieldH/2 + self.fieldH/4*random.random()),
            (self.W - (self.sideLength), self.H/2),
        ]

        # Add robots
        self.robots = [Robot(spot,1,i) for i,spot in enumerate(self.robotSpots[:self.maxPlayers]) if i < self.nPlayers] \
                      + [Robot(spot,-1,self.nPlayers+i) for i,spot in enumerate(self.robotSpots[self.maxPlayers:]) if i < self.nPlayers]
        for robot in self.robots:
            self.space.add(robot.leftFoot.body,robot.leftFoot,robot.rightFoot.body,robot.rightFoot,robot.joint,robot.rotJoint)

        # Add ball
        self.ball = Ball(self.W/2,self.H/2,self.ballRadius)
        self.space.add(self.ball.shape.body,self.ball.shape)

        # Add goalposts
        self.goalposts = [
            Goalpost(self.sideLength,self.H/2+self.goalWidth,self.goalPostRadius),
            Goalpost(self.sideLength,self.H/2-self.goalWidth,self.goalPostRadius),
            Goalpost(self.W-self.sideLength,self.H/2+self.goalWidth,self.goalPostRadius),
            Goalpost(self.W-self.sideLength,self.H/2-self.goalWidth,self.goalPostRadius),
        ]
        for goal in self.goalposts:
            self.space.add(goal.shape.body,goal.shape)

        # Handle robot-goalpost collision
        h = self.space.add_collision_handler(
            CollisionType.Robot,
            CollisionType.Goalpost)
        h.post_solve = self.goalpostCollision
        h.separate = self.separate

        # Handle robot-robot collision
        h = self.space.add_collision_handler(
            CollisionType.Robot,
            CollisionType.Robot)
        h.begin = self.robotPushingDet
        h.post_solve = self.robotCollision
        h.separate = self.separate

        # Handle robot-ball collision
        h = self.space.add_collision_handler(
            CollisionType.Robot,
            CollisionType.Ball)
        h.begin = self.ballCollision

        # Render options
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Robot Soccer")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    # Called when robots begin touching
    def robotPushingDet(self,arbiter, space, data):

        # Get objects involved
        robot1 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        robot2 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[1] or robot.rightFoot == arbiter.shapes[1]))

        print("Robot Collision")

        # Get velocities and positions
        v1 = arbiter.shapes[0].body.velocity
        v2 = arbiter.shapes[1].body.velocity
        p1 = robot1.getPos()
        p2 = robot2.getPos()
        dp = p1-p2

        # Robot might be pushing if it's walking towards the other
        robot1.mightPush = v1.length > 1 and math.cos(dp.angle-v1.angle) < -0.4
        robot2.mightPush = v2.length > 1 and math.cos(dp.angle-v2.angle) > 0.4

        # Set touching and touch counter variables
        robot1.touching = True
        robot2.touching = True
        robot1.touchCntr = 0
        robot2.touchCntr = 0

        return True

    # Called when robots are touching (before collision is computed)
    def robotCollision(self,arbiter, space, data):

        # Get objects involved
        robot1 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        robot2 = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[1] or robot.rightFoot == arbiter.shapes[1]))

        # The two legs might collide
        if robot1 == robot2:
            return

        # Increment touching
        if not (robot1.fallen or robot1.penalized):
            robot1.touchCntr += 1
        if not (robot2.fallen or robot2.penalized):
            robot2.touchCntr += 1

        # Compute fall probability thresholds - the longer the robots are touching the more likely they will fall
        normalThresh = 0.99995
        pushingThresh = 0.99998

        # Determine if robots fall
        r = random.random()
        if r > (pushingThresh if robot1.mightPush else normalThresh)**robot1.touchCntr and not robot1.fallen:
            self.fall(robot1)
            robot1.touchCntr = 0
        r = random.random()
        if r > (pushingThresh if robot2.mightPush else normalThresh)**robot2.touchCntr and not robot2.fallen:
            self.fall(robot2)
            robot2.touchCntr = 0

        # Penalize robots for pushing
        if robot1.mightPush and not robot2.mightPush and robot2.fallen and robot1.team!=robot2.team:
            print("Robot 1 Pushing")
            self.penalize(robot1)
            robot1.touchCntr = 0
        elif robot2.mightPush and not robot1.mightPush and robot1.fallen and robot1.team!=robot2.team:
            print("Robot 2 Pushing")
            self.penalize(robot2)
            robot2.touchCntr = 0


    # Called when robots stop touching
    def separate(self,arbiter, space, data):

        print("Collision Separated")

        # Get robot
        robots = [robot for robot in self.robots if (robot.leftFoot in arbiter.shapes or robot.rightFoot in arbiter.shapes)]

        # Reset collision variables
        for robot in robots:
            robot.touching = False
            robot.mightPush = False
            robot.touchCntr = 0


    # Called when robot collides with goalpost
    def goalpostCollision(self,arbiter, space, data):

        # Get robot
        robot = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))

        # Don't make a fallen robot fall again from touching the post
        if robot.fallen:
            robot.touchCntr = 0
            return

        # Set things on first run
        if not robot.touching:
            print("Goalpost Collision")
            robot.touching = True
            robot.touchCntr = 0

        # Increment touch counter and compute fall probability threshold
        robot.touchCntr += 1
        pushingThresh = 0.9999 ** robot.touchCntr

        # Determine if robot falls
        r = random.random()
        if r > pushingThresh:
            self.fall(robot)


    # Called when robot touches the ball
    def ballCollision(self,arbiter, space, data):

        # Get robot
        robot = next(robot for robot in self.robots if (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))

        if self.ballOwned != 0:
            if robot.team != self.ballOwned and not robot.penalized:
                self.penalize(robot)
            else:
                print("Ball Free")
                self.ballOwned = -1
                self.gracePeriod = 0
                self.ballFreeCntr = 0

        # Shift lastKicked array
        self.ball.lastKicked = [robot.id] + self.ball.lastKicked
        if len(self.ball.lastKicked) > 4:
            self.ball.lastKicked = self.ball.lastKicked[:4]

        return True

    # Drawing
    def drawStaticObjects(self):

        # Field green
        self.screen.fill((0, 255, 0))

        for line in self.lines:
            pygame.draw.line(self.screen,(255,255,255),line[0],line[1],self.lineWidth)

        for cross in self.fieldCrosses:
            pygame.draw.circle(self.screen,(255,255,255),cross[0],cross[1]*2,0)

        pygame.draw.circle(self.screen, (255,255,255), self.centerCircle[0], self.centerCircle[1] * 2, self.lineWidth)

        self.space.debug_draw(self.draw_options)

    # Ball free kick
    def ballFreeKickProcess(self,team):
        if team == 0:
            time = 1000 / self.timeStep
            if self.gracePeriod > 0:
                self.gracePeriod -= time
                if self.gracePeriod < 0:
                    print("Grace period over")
                    self.gracePeriod = 0
                    self.ballFreeCntr = 9999
            elif self.ballFreeCntr > 0:
                self.ballFreeCntr -= time
                if self.ballFreeCntr < 0:
                    print("Ball Free")
                    self.ballFreeCntr = 0
                    self.ballOwned = 0
        else:
            print("Free kick", team)
            self.ballOwned = team
            self.gracePeriod = 14999
            self.ballFreeCntr = 0


    # Detect ball movements
    def isBallOutOfField(self):

        # Setup basic parameters
        finished = False
        pos = self.ball.shape.body.position

        # Current tem rewards
        currReward = [0,0]

        # Field edges
        outMin = self.sideLength - self.ballRadius
        outMaxX = self.W - self.sideLength + self.ballRadius
        outMaxY = self.H -self.sideLength + self.ballRadius

        # Team to award free kick
        team = 0

        # If ball is out
        if pos.y < outMin or pos.x < outMin or pos.y > outMaxY or pos.x > outMaxX:
            x = self.W/2
            y = self.H/2

            # If out on the sides
            team = self.robots[self.ball.lastKicked[0]].team
            if pos.y < outMin or pos.y > outMaxY:
                x = pos.x + 50 if team < 0 else pos.x - 50
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
                        if team < 0:
                            x = self.sideLength + self.penaltyLength
                        # Corner
                        else:
                            x = self.sideLength
                            y = self.sideLength if pos.y < self.H/2 else self.H-self.sideLength
                    else:
                        # Kick out
                        if team > 0:
                            x = self.W - (self.sideLength + self.penaltyLength)
                        # Corner
                        else:
                            x = self.W - self.sideLength
                            y = self.sideLength if pos.y < self.H/2 else self.H-self.sideLength

            # Move ball to middle and stop it
            self.ball.shape.body.position = pymunk.Vec2d(x,y)
            self.ball.shape.body.velocity = pymunk.Vec2d(0.0,0.0)
            self.ball.shape.body.angular_velocity = 0.0

        # Update free kick status
        self.ballFreeKickProcess(-team)

        # Add ball movement to the reward
        if not finished:
            currReward[0] += self.ball.shape.body.position.x - self.ball.prevPos.x
            currReward[1] -= self.ball.shape.body.position.x - self.ball.prevPos.x

        # Update previous position
        self.ball.prevPos = self.ball.shape.body.position

        # Create discounted personal rewards for the robots involved
        for i, id in enumerate(self.ball.lastKicked):
            self.robotRewards[id] += currReward[0] * (self.kickDiscount ** i) if id < self.nPlayers else currReward[1] * (self.kickDiscount ** i)

        # Create personal rewards for nearby robots not touching the ball, but only negative rewards
        for robot in self.robots:
            if (robot.getPos() - pos).length < 150:
                if not robot.penalized and self.ballOwned != robot.team and self.ballFreeCntr > 0:
                    print("Illegal position", robot.id, robot.team)
                    self.penalize(robot)
                if robot.id not in self.ball.lastKicked:
                    self.robotRewards[int(robot.id)] += min(0, currReward[0] * self.kickDiscount if robot.id < self.nPlayers else
                    currReward[1] * self.kickDiscount)

        # Update team rewards
        self.teamRewards[0] += currReward[0]
        self.teamRewards[1] += currReward[1]

        return finished

    # Robot falling
    def fall(self,robot):
        print("Fall", robot.fallCntr, robot.team)

        # Get robot position
        pos = robot.getPos()

        # Find objects nearby
        filter = pymunk.shape_filter.ShapeFilter(categories=0b101)
        shapes = self.space.point_query(pos,40,filter)

        # Punish robots for falling
        self.robotRewards[robot.id] -= 100

        # For closeby objects (that are not the robot)
        for query in shapes:
            if query.shape != robot.leftFoot and query.shape != robot.rightFoot:

                # Compute force of the robot's fall
                force = robot.velocity*robot.leftFoot.body.mass*query.shape.body.mass/50.0

                # Get direction of the force
                dp = pos - query.shape.body.position

                # Force is proportional to the distance
                dp = -dp*force/dp.length

                # Apply force
                query.shape.body.apply_force_at_world_point(dp,pos)

                # If the fallen robots touched the ball, update the last touched variable
                if query.shape == self.ball.shape:
                    self.ball.lastKicked = [robot.id] + self.ball.lastKicked
                    if len(self.ball.lastKicked) > 4:
                        self.ball.lastKicked = self.ball.lastKicked[:4]
                    if self.ballOwned != 0:
                        print("Ball Free")
                        self.gracePeriod = 0
                        self.ballFreeCntr = 0
                        self.ballOwned = 0

        # Update the color of the fallen robot
        robot.leftFoot.color = (255, int(75*(1+robot.team)), int(75*(1-robot.team)))
        robot.rightFoot.color = (255, int(75*(1+robot.team)), int(75*(1-robot.team)))

        # Set variables
        robot.fallen = True

        # Set number of falls and getup time
        robot.fallCntr += 1
        robot.moveTime = 3000

        # If the robot fell 3 times, penalize
        if robot.fallCntr > 2:
            print("Fallen robot", robot.fallCntr, robot.team)
            self.penalize(robot)

    # Penlize robot
    def penalize(self,robot):

        # Update penalized status
        robot.penalized = True
        teamIdx = 0 if robot.team > 0 else 1
        robot.penalTime = self.penalTimes[teamIdx]

        # Punish the robot
        self.robotRewards[robot.id] -= self.penalTimes[teamIdx]/100

        # Increase penalty time for team
        self.penalTimes[teamIdx] += 5000

        # Compute robot position
        pos = robot.getPos()
        xOffs = (robot.id - teamIdx*self.nPlayers) * 40
        x = self.W - (self.sideLength + self.penaltyLength + xOffs) if robot.team < 0 else self.sideLength + self.penaltyLength + xOffs
        print(robot.id,xOffs,x)
        y = self.sideLength if self.ball.shape.body.position.y < self.H/2 else self.H-self.sideLength

        # Move feet
        robot.leftFoot.body.position = pymunk.Vec2d(x, y)
        robot.rightFoot.body.position = pymunk.Vec2d(x, y)

        # Stop feet, and change color
        robot.leftFoot.body.angle = math.pi / 2 if y < self.H/2 else -math.pi / 2
        robot.leftFoot.body.velocity = pymunk.Vec2d(0.0, 0.0)
        robot.leftFoot.body.angular_velocity = 0.0
        robot.leftFoot.color = (255, 0, 0)
        robot.rightFoot.body.angle = math.pi / 2 if y < self.H/2 else -math.pi / 2
        robot.rightFoot.body.velocity = pymunk.Vec2d(0.0, 0.0)
        robot.rightFoot.body.angular_velocity = 0.0
        robot.rightFoot.color = (255, 0, 0)

        # Set moving variables
        if robot.kicking:
            robot.kicking = False
            # If the robot was kicking, the joint between its legs was removed. It needs to be added back
            self.space.add(robot.joint)

    # Robot update function
    def tick(self,robot):

        # Get timestep
        time = 1000 / self.timeStep

        # If the robot is moving
        if robot.moveTime > 0:
            robot.moveTime -= time

            # Move head
            if robot.headMoving != 0:
                robot.headAngle += robot.headMoving
                robot.headAngle = max(-robot.headMaxAngle,min(robot.headMaxAngle,robot.headAngle))

            # Update kicking
            if robot.kicking:
                # Get which foot
                foot = robot.rightFoot if robot.foot else robot.leftFoot

                # 500 ms into the kick the actual movement starts
                if robot.moveTime + time > 500 and robot.moveTime <= 500:
                    # Remove joint between legs to allow the leg to move independently
                    self.space.remove(robot.joint)

                    # Set velocity
                    velocity = pymunk.Vec2d(robot.velocity * 2.5, 0)
                    angle = foot.body.angle
                    velocity.rotate(angle)
                    foot.body.velocity = velocity

                # 600 ms into the kick, the legs starts moving back
                if robot.moveTime + time > 400 and robot.moveTime <= 400:
                    # Set velocity
                    velocity = pymunk.Vec2d(robot.velocity * 2.5, 0)
                    angle = foot.body.angle
                    velocity.rotate(angle)
                    foot.body.velocity = -velocity

                # 700 ms into the kick the leg stops moving and returns to its original position
                elif robot.moveTime <= 300:
                    # Stop leg and return it to its starting position
                    foot.body.velocity = pymunk.Vec2d(0,0)
                    robot.kicking = False
                    foot.body.position = robot.initPos

                    # Add joint back
                    self.space.add(robot.joint)

            # If the movement is over
            if robot.moveTime <= 0:

                # Stop the robot
                robot.moveTime = 0
                robot.headMoving = 0
                robot.leftFoot.body.velocity = pymunk.Vec2d(0,0)
                robot.leftFoot.body.angular_velocity = 0.0
                robot.rightFoot.body.velocity = pymunk.Vec2d(0,0)
                robot.rightFoot.body.angular_velocity = 0.0

                # If the robot fell
                if robot.fallen:

                    # Is might fall again
                    r = random.random()
                    if r > 0.9:
                        self.fall(robot)
                        return

                    print("Getup", robot.team)

                    # Reset color and variables
                    robot.leftFoot.color = (255, int(127*(1-robot.team)), int(127*(1+robot.team)))
                    robot.rightFoot.color = (255, int(127*(1-robot.team)), int(127*(1+robot.team)))
                    robot.fallen = False
                    robot.fallCntr = 0

        # Handle penalties
        if robot.penalized:
            robot.penalTime -= time

            # If expired
            if robot.penalTime <= 0:
                print("Unpenalized")

                # Reset all variables
                robot.penalTime = 0
                robot.penalized = False
                robot.fallCntr = 0
                robot.fallen = False

                ballPos = self.ball.shape.body.position

                # Put the robot on the right side of the field
                pos = robot.leftFoot.body.position
                pos.y = self.sideLength if ballPos.y > self.H/2 else self.H-self.sideLength
                robot.leftFoot.body.angle = math.pi / 2 if ballPos.y > self.H/2 else -math.pi / 2
                robot.leftFoot.body.position = pos
                robot.leftFoot.color = (255, int(127*(1-robot.team)), int(127*(1+robot.team)))
                pos = robot.rightFoot.body.position
                pos.y = self.sideLength if ballPos.y > self.H/2 else self.H-self.sideLength
                robot.rightFoot.body.angle = math.pi / 2 if ballPos.y > self.H/2 else -math.pi / 2
                robot.rightFoot.body.position = pos
                robot.rightFoot.color = (255, int(127*(1-robot.team)), int(127*(1+robot.team)))
        else:
            teamIdx = 0 if robot.team > 0 else 1

            #Get robot and penalty positions
            pos = robot.getPos()
            robX = self.W-pos.x if teamIdx else pos.x
            penX = self.sideLength + self.penaltyLength + self.lineWidth/2

            # If robot is in the penalty box
            if robX < penX and pos.y > (self.H/2-self.penaltyWidth) and pos.y < (self.H/2+self.penaltyWidth):

                # If not in the defenders, add it or penalize if limit is reached
                if robot.id not in self.defenders[teamIdx]:
                    if len(self.defenders[teamIdx]) >= 2:
                        print("Illegal defender")
                        self.penalize(robot)
                    else:
                        self.defenders[teamIdx].append(robot.id)

            # If the robot is not in the box anymore, remove it from the defender list
            elif robot.id in self.defenders[teamIdx]:
                self.defenders[teamIdx].remove(robot.id)

    # Robo leaving field detection
    def isLeavingField(self,robot):

        # Robot position
        pos = robot.getPos()

        # Get field edges
        outMin = 5
        outMaxX = self.W-5
        outMaxY = self.H-5

        # If robot is about to leave, penalize
        if pos.y < outMin or pos.x < outMin or pos.y > outMaxY or pos.x > outMaxX:
            self.penalize(robot)

    # Main step function
    def step(self,actions):
        t1 = time.clock()

        # Setup reward and state variables
        self.teamRewards = [0,0]
        self.robotRewards = [0,0]*self.nPlayers
        observations = []
        finished = False

        # Run simulation for 500 ms (time for every action, except the kick)
        for i in range(50):

            # Draw lines
            if self.render:
                self.drawStaticObjects()

            # Sanity check
            if len(actions) != len(self.robots):
                print("Error: There must be action s for every robot")
                exit(0)

            # Robot loop
            for action, robot in zip(actions,self.robots):
                # Apply action as first step
                if i == 0:
                    self.processAction(action,robot)

                # Update robots
                self.tick(robot)
                self.isLeavingField(robot)

            # Process ball position
            finished = self.isBallOutOfField()

            # Run simulation
            self.space.step(1 / self.timeStep)

            # Get observations every 100 ms
            if i % 10 == 9:
                if self.observationType == ObservationType.Full:
                    observations.append([self.getFullState(robot) for robot in self.robots])
                else:
                    observations.append([self.getRobotVision(robot) for robot in self.robots])

            # Render
            if self.render:
                pygame.display.flip()
                self.clock.tick(self.timeStep)
                cv2.waitKey(1)

        t2 = time.clock()
        print((t2-t1)*1000)

        return self.getFullState(),observations,self.teamRewards,self.robotRewards,finished

    # Action handler
    def processAction(self, action, robot):

        # Get 4 action types
        move,turn,head,kick = action

        # Don't allow movement or falling unless no action is being performed
        canMove = not(robot.penalized or robot.kicking or robot.fallen)

        # Moving has a small chance of falling
        if move > 0 and canMove:
            r = random.random()
            if r > 0.995:
                self.fall(robot)
                return
            robot.step(move-1)

        # Turning has a small chance of falling
        if turn > 0 and canMove:
            r = random.random()
            if r > 0.995:
                self.fall(robot)
                return
            robot.turn(turn-1)

        # Head movements have no chance of falling
        if head:
            robot.turnHead(head)

        # Kick has a higher chance of falling. Also, kick cannot be performed together with any other motion
        if kick > 0 and move == 0 and turn == 0 and canMove:
            r = random.random()
            if r > 0.95:
                self.fall(robot)
                return
            robot.kick(kick-1)

    # Get true object state for a robot
    def getFullState(self,robot=None):
        if robot is None:
            state = [[self.ball.shape.body.position,self.ballOwned]] + [[rob.getPos(),rob.team,rob.fallen or rob.penalized] for rob in self.robots]
        else:
            state = [[self.ball.shape.body.position,self.ballOwned*robot.team]] + \
                   [[robot.getPos(),1,robot.fallen or robot.penalized]] +\
                   [[rob.getPos(),rob.team*robot.team,rob.fallen or rob.penalized] for rob in self.robots if rob != robot]

            for elem in state:
                elem[0].x *= robot.team

        return state

    # Getting vision
    def getRobotVision(self,robot):

        #Get position and orientation
        pos = robot.getPos()
        angle = robot.leftFoot.body.angle
        headAngle = angle+robot.headAngle

        # FoV
        angle1 = headAngle+robot.fieldOfView
        angle2 = headAngle-robot.fieldOfView

        # Edge of field of view
        vec1 = pymunk.Vec2d(1,0)
        vec1.rotate(angle1)

        # Other edge of field of view
        vec2 = pymunk.Vec2d(1,0)
        vec2.rotate(angle2)

        # Check if objects are seen
        ballDets = [isSeenInArea(self.ball.shape.body.position - pos,vec1,vec2,self.maxVisDist[0],headAngle,self.ballRadius*2)+ [self.ballOwned*robot.team]]
        robDets = [isSeenInArea(rob.getPos() - pos,vec1,vec2,self.maxVisDist[1],headAngle,Robot.totalRadius)+[robot.team == rob.team,robot.fallen or robot.penalized] for rob in self.robots if robot != rob]
        goalDets = [isSeenInArea(goal.shape.body.position - pos,vec1,vec2,self.maxVisDist[1],headAngle,self.goalPostRadius*2) for goal in self.goalposts]
        crossDets = [isSeenInArea(cross[0] - pos,vec1,vec2,self.maxVisDist[0],headAngle,self.penaltyRadius*2) for cross in self.fieldCrosses]
        lineDets = [isLineInArea(p1 - pos,p2 - pos,vec1,vec2,self.maxVisDist[1],headAngle) for p1,p2 in self.lines]
        circleDets = isSeenInArea(self.centerCircle[0] - pos,vec1,vec2,self.maxVisDist[1],headAngle,self.centerCircleRadius*2,False)

        # Get interactions between certain object classes
        robRobInter = [max([doesInteract(rob1[1],rob2[1],Robot.totalRadius*2) for rob1 in robDets if rob1 != rob2]) for rob2 in robDets] if self.nPlayers > 1 else [0]
        robBallInter = max([doesInteract(rob[1],ballDets[0][1],Robot.totalRadius*2) for rob in robDets])
        robPostInter = [max([doesInteract(rob[1],post[1],Robot.totalRadius*2) for rob in robDets]) for post in goalDets]
        robCrossInter = [max([doesInteract(rob[1],cross[1],Robot.totalRadius*2) for rob in robDets]) for cross in crossDets]
        ballPostInter = max([doesInteract(ballDets[0][1],post[1],self.ballRadius*8,False) for post in goalDets])
        ballCrossInter = [doesInteract(ballDets[0][1],cross[1],self.ballRadius*4,False)for cross in crossDets]

        # Random position noise and false negatives
        [addNoise(ball, self.noiseType, max(robBallInter,ballPostInter), self.noiseMagnitude, self.randBase, self.maxVisDist[0], True) for ball in ballDets]
        [addNoise(rob, self.noiseType, robRobInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i,rob in enumerate(robDets)]
        [addNoise(goal, self.noiseType, robPostInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i,goal in enumerate(goalDets)]
        [addNoise(cross, self.noiseType, max(robCrossInter[i], ballCrossInter[i]), self.noiseMagnitude, self.randBase, self.maxVisDist[0], True) for i,cross in enumerate(crossDets)]
        addNoise(circleDets, self.noiseType, 0, self.noiseMagnitude, self.randBase, self.maxVisDist[1])
        [addNoiseLine(line, self.noiseType, self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i,line in enumerate(lineDets)]

        # Balls and crosses might by miscalssified - move them in the other list
        for ball in ballDets:
            if ball[0] == SightingType.Misclassified:
                crossDets.append((SightingType.Normal,ball[1],ball[2]))
        for cross in crossDets:
            if cross[0] == SightingType.Misclassified:
                ballDets.append((SightingType.Normal,cross[1],cross[2]))

        # Random false positives
        for i in range(10):
            if random.random() < self.randBase:
                c = random.randint(0,5)
                d = random.random()*self.maxVisDist[1]
                a = random.random()*2*robot.fieldOfView - robot.fieldOfView
                pos = pymunk.Vec2d(d,0)
                pos.rotate(a)
                if c == 0:
                    ballDets.insert(len(ballDets),
                                   [SightingType.Normal,pos,self.ballRadius*2*(1-0.4*(random.random()-0.5))])
                elif c == 1:
                    robDets.insert(len(robDets),
                                   [SightingType.Normal,pos,Robot.totalRadius*(1-0.4*(random.random()-0.5)),random.random() > 0.5,random.random() > 0.75])
                elif c == 2:
                    goalDets.insert(len(goalDets),
                                   [SightingType.Normal,pos,self.goalPostRadius*2*(1-0.4*(random.random()-0.5))])
                elif c == 3:
                    crossDets.insert(len(crossDets),
                                   [SightingType.Normal,pos,self.penaltyRadius*2*(1-0.4*(random.random()-0.5))])

        # FP Balls near robots
        if self.noiseType == NoiseType.Realistic:
            for rob in robDets:
                if rob[0] == SightingType.Normal and random.random() < self.randBase*10 and rob[1].length < 250:
                    if random.random() < self.randBase*8:
                        rob[0] = SightingType.NoSighting
                    offset = pymunk.Vec2d(2*random.random()-1.0,2*random.random()-1.0)*Robot.totalRadius
                    ballDets.insert(len(ballDets),
                                   [SightingType.Normal,rob[1]+offset,self.ballRadius*2*(1-0.4*(random.random()-0.5))])


        # Remove occlusion and misclassified originals
        ballDets = [ball for i,ball in enumerate(ballDets) if ball[0] != SightingType.NoSighting and ball[0] != SightingType.Misclassified]
        robDets = [rob for i,rob in enumerate(robDets) if rob[0] != SightingType.NoSighting]
        goalDets = [goal for i,goal in enumerate(goalDets) if goal[0] != SightingType.NoSighting]
        crossDets = [cross for i,cross in enumerate(crossDets) if cross[0] != SightingType.NoSighting and cross[0] != SightingType.Misclassified]
        lineDets = [line for i,line in enumerate(lineDets) if line[0] != SightingType.NoSighting]

        if self.observationType == ObservationType.Image:

            # Initialize images
            bottomCamImg = np.zeros((480,640))
            topCamImg = np.zeros((480,640))

            for line in lineDets:

                # Points to transform: [start, start+thickness, end]
                linevec = np.array([
                    [(-line[1].y-line[2].y)/2,                  0,  (line[1].x+line[2].x)/2,  1],
                    [(-line[1].y-line[2].y)/2+self.lineWidth/2, 0,  (line[1].x+line[2].x)/2,  1],
                    [-line[1].y,                                0,  line[1].x,                1],
                    [-line[2].y,                                0,  line[2].x,                1]
                ]).transpose()

                # Project points and estimate radius (projected size of line thickness)
                tProj,tRad,bProj,bRad = projectPoints(linevec)

                # Draw
                cv2.line(topCamImg,     (int(tProj[0,2]),int(tProj[1,2])),  (int(tProj[0,3]),int(tProj[1,3])), 4, tRad)
                cv2.line(bottomCamImg,  (int(bProj[0,2]),int(bProj[1,2])),  (int(bProj[0,3]),int(bProj[1,3])), 4, bRad)

            if circleDets[0] != SightingType.NoSighting:

                # Rotated directional vector
                ellipseOffs = pymunk.Vec2d(circleDets[2],0)
                ellipseOffs.rotate(math.pi/4)

                # Points to transform: [center, 6 more points on the circle]
                circlevec = np.array([
                    [-circleDets[1].y,                0,    circleDets[1].x,                1],
                    [-circleDets[1].y,                0,    circleDets[1].x-circleDets[2],  1],
                    [-circleDets[1].y,                0,    circleDets[1].x+circleDets[2],  1],
                    [-circleDets[1].y-circleDets[2],  0,    circleDets[1].x,                1],
                    [-circleDets[1].y+circleDets[2],  0,    circleDets[1].x,                1],
                    [-circleDets[1].y-ellipseOffs.y,  0,    circleDets[1].x+ellipseOffs.x,  1],
                    [-circleDets[1].y+ellipseOffs.x,  0,    circleDets[1].x-ellipseOffs.y,  1],
                    [-circleDets[1].y+ellipseOffs.y,  0,    circleDets[1].x+ellipseOffs.x,  1],
                    [-circleDets[1].y-ellipseOffs.x,  0,    circleDets[1].x-ellipseOffs.y,  1],
                ]).transpose()

                # Project points and estimate radius (projected size of line thickness)
                tProj,tRad,bProj,bRad = projectPoints(circlevec,False)

                # estimate line thickness from center distance
                tThickness = 15-max(0,min(14,int(circleDets[1].length/40)))
                bThickness = 30-max(0,min(29,int(circleDets[1].length/20)))

                # Estimate conic parameters
                tParams = estimateConic(tProj[:,1:]-tProj[:,0:1])
                bParams = estimateConic(bProj[:,1:]-bProj[:,0:1])

                # Get [x,y] coordinates of the conic for y in [0,480)
                # x1 and x2 are the two curves that make up the conic (they might be separate due to field of vision)
                tx1, tx2 = getConicPoints(480, tProj[:,0], tParams)
                bx1, bx2 = getConicPoints(480, bProj[:,0], bParams)

                # Draw polygon on points
                cv2.polylines(topCamImg, [tx1], False, 4, tThickness)
                cv2.polylines(topCamImg, [tx2], False, 4, tThickness)
                cv2.polylines(bottomCamImg, [bx1], False, 4, bThickness)
                cv2.polylines(bottomCamImg, [bx2], False, 4, bThickness)

                # Connect the first and last elements of the two curves, unless they are at the edges of the images
                if tx1.shape[0]:
                    if tx1[0, 1]:
                        cv2.line(topCamImg, tuple(tx1[0]), tuple(tx2[0]), 4, tThickness)
                    if tx1[-1, 1] < 480 - 1:
                        cv2.line(topCamImg, tuple(tx1[-1]), tuple(tx2[-1]), 4, tThickness)
                if bx1.shape[0]:
                    if bx1[0, 1]:
                        cv2.line(bottomCamImg, tuple(bx1[0]), tuple(bx2[0]), 4, bThickness)
                    if bx1[-1, 1] < 480 - 1:
                        cv2.line(bottomCamImg, tuple(bx1[-1]), tuple(bx2[-1]), 4, bThickness)

            for rob in robDets:

                # Points to transform: [bottom left, bottom right, top left, top right]
                robvec = np.array([
                    [-rob[1].y-rob[2],  0,  rob[1].x,   1],
                    [-rob[1].y+rob[2],  58, rob[1].x,   1]
                ]).transpose()

                # Project points (without radius estimation)
                tProj,tRad,bProj,bRad = projectPoints(robvec,False)

                # Draw
                cv2.rectangle(topCamImg,    (int(tProj[0,0]),int(tProj[1,0])),  (int(tProj[0,1]),int(tProj[1,1])), 2, -1)
                cv2.rectangle(bottomCamImg, (int(bProj[0,0]),int(bProj[1,0])),  (int(bProj[0,1]),int(bProj[1,1])), 2, -1)

            for goal in goalDets:

                # Points to transform: [bottom, bottom+thickness, top]
                goalvec = np.array([
                    [-goal[1].y,            0,  goal[1].x,  1],
                    [-goal[1].y+goal[2]/2,  0,  goal[1].x,  1],
                    [-goal[1].y,            80, goal[1].x,  1]
                ]).transpose()

                # Project points and estimate radius (projected size of goal thickness)
                tProj,tRad,bProj,bRad = projectPoints(goalvec)

                # Draw
                cv2.line(topCamImg,     (int(tProj[0,0]),int(tProj[1,0])),  (int(tProj[0,2]),int(tProj[1,2])), 3, tRad)
                cv2.line(bottomCamImg,  (int(bProj[0,0]),int(bProj[1,0])),  (int(bProj[0,2]),int(bProj[1,2])), 3, bRad)

            for cross in crossDets:

                # Points to transform: [center, center+thickness]
                crossvec = np.array([
                    [-cross[1].y,               0,  cross[1].x, 1],
                    [-cross[1].y+cross[2]/2,    0,  cross[1].x, 1]
                ]).transpose()

                # Project points and estimate radius (projected size of cross radius)
                tProj,tRad,bProj,bRad = projectPoints(crossvec)

                # Draw
                cv2.circle(topCamImg,       (int(tProj[0,0]),int(tProj[1,0])), tRad, 4, -1)
                cv2.circle(bottomCamImg,    (int(bProj[0,0]),int(bProj[1,0])), bRad, 4, -1)

            for ball in ballDets:

                # Points to transform: [center, center+thickness]
                ballvec = np.array([
                    [-ball[1].y,            ball[2]/2,  ball[1].x,  1],
                    [-ball[1].y+ball[2]/2,  ball[2]/2,  ball[1].x,  1]
                ]).transpose()

                # Project points and estimate radius (projected size of ball radius)
                tProj,tRad,bProj,bRad = projectPoints(ballvec)

                # Draw
                cv2.circle(topCamImg,       (int(tProj[0,0]),int(tProj[1,0])),  tRad, 1, -1)
                cv2.circle(bottomCamImg,    (int(bProj[0,0]),int(bProj[1,0])),  bRad, 1, -1)

        if self.render and robot.id == self.visId:

            # Visualization image size
            H = self.W//2-50
            W = self.W//2
            xOffs = 150
            img = np.zeros((H*2,W*2,3)).astype('uint8')

            # Rotate FoV back for visualization
            vec1.rotate(-headAngle)
            vec2.rotate(-headAngle)

            # Draw
            cv2.line(img,(xOffs,H),(int(xOffs+vec1.x*1000),int(H-vec1.y*1000)),(255,255,0))
            cv2.line(img,(xOffs,H),(int(xOffs+vec2.x*1000),int(H-vec2.y*1000)),(255,255,0))

            # Draw all objects
            # Partially seen and distant objects are dim
            # Objects are drawn from the robot center
            for line in lineDets:
                color = (255,255,255) if line[0] == SightingType.Normal else (127,127,127)
                cv2.line(img,(int(xOffs+line[1].x),int(-line[1].y+H)),(int(xOffs+line[2].x),int(-line[2].y+H)),color,self.lineWidth)

            if circleDets[0] != SightingType.NoSighting:
                color = (255,0,255) if circleDets[0] == SightingType.Normal else (127,0,127)
                cv2.circle(img,(int(xOffs+circleDets[1].x),int(-circleDets[1].y+H)),int(circleDets[2]),color,self.lineWidth)

            for cross in crossDets:
                color = (255,255,255) if cross[0] == SightingType.Normal else (127,127,127)
                cv2.circle(img,(int(xOffs+cross[1].x),int(-cross[1].y+H)),int(cross[2]),color,-1)

            for goal in goalDets:
                color = (255,0,0) if goal[0] == SightingType.Normal else (127,0,0)
                cv2.circle(img,(int(xOffs+goal[1].x),int(-goal[1].y+H)),int(goal[2]),color,-1)

            for i,rob in enumerate(robDets):
                color = (0,255,0) if rob[0] == SightingType.Normal else (0,127,0)
                cv2.circle(img,(int(xOffs+rob[1].x),int(-rob[1].y+H)),int(rob[2]),color,-1)

            for ball in ballDets:
                color = (0,0,255) if ball[0] == SightingType.Normal else (0,0,127)
                cv2.circle(img,(int(xOffs+ball[1].x),int(-ball[1].y+H)),int(ball[2]),color,-1)

            cv2.imshow(("Robot %d" % robot.id),img)
            if self.observationType == ObservationType.Image:
                cv2.imshow("Bottom",colorize(bottomCamImg))
                cv2.imshow("Top",colorize(topCamImg))

        if self.observationType == ObservationType.Image:
            return topCamImg,bottomCamImg
        return ballDets,robDets,goalDets,crossDets,lineDets,circleDets
