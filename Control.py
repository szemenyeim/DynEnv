import pygame
import pymunk.pygame_util
import time
from Ball import *
from Goalpost import *
from Robot import *
from cutils import *
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
        self.visId = random.randint(0,self.nPlayers*2-1)

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
        self.randBase = 0.01 * noiseMagnitude
        self.noiseMagnitude = noiseMagnitude
        self.maxVisDist = [self.W*0.4,self.W*0.8]

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
        self.robots = [Robot(spot,0,i) for i,spot in enumerate(self.robotSpots[:self.maxPlayers]) if i < self.nPlayers] \
                      + [Robot(spot,1,self.nPlayers+i) for i,spot in enumerate(self.robotSpots[self.maxPlayers:]) if i < self.nPlayers]
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
        robot1.touchCntr += 1
        robot2.touchCntr += 1

        # Compute fall probability thresholds - the longer the robots are touching the more likely they will fall
        normalThresh = 0.99995 ** robot1.touchCntr
        pushingThresh = 0.99998 ** robot1.touchCntr

        # Determine if robots fall
        r = random.random()
        if r > (pushingThresh if robot1.mightPush else normalThresh) and not robot1.fallen:
            self.fall(robot1)
            robot1.touchCntr = 0
        r = random.random()
        if r > (pushingThresh if robot2.mightPush else normalThresh) and not robot2.fallen:
            self.fall(robot2)
            robot2.touchCntr = 0

        # Penalize robots for pushing
        if robot1.mightPush and not robot2.mightPush and robot2.fallen:
            print("Robot 1 Pushing")
            self.penalize(robot1)
            robot1.touchCntr = 0
        elif robot2.mightPush and not robot1.mightPush and robot1.fallen:
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
            currReward[0] += self.ball.shape.body.position.x - self.ball.prevPos.x
            currReward[1] -= self.ball.shape.body.position.x - self.ball.prevPos.x

        # Update previous position
        self.ball.prevPos = self.ball.shape.body.position

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

        # Update the color of the fallen robot
        robot.leftFoot.color = (255, int(100*(1-robot.team)), int(100*robot.team))
        robot.rightFoot.color = (255, int(100*(1-robot.team)), int(100*robot.team))

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
        print("Penalized")

        # Update penalized status
        robot.penalized = True
        robot.penalTime = self.penalTimes[robot.team]

        # Punish the robot
        self.robotRewards[robot.id] -= self.penalTimes[robot.team]/100

        # Increase penalty time for team
        self.penalTimes[robot.team] += 5000

        # Compute robot position
        pos = robot.getPos()
        x = self.sideLength + self.penaltyLength if robot.team else self.W - (self.sideLength + self.penaltyLength)
        y = self.sideLength if pos.y < self.H/2 else self.H-self.sideLength

        # Move feet
        robot.leftFoot.body.position = pymunk.Vec2d(x-10, y)
        robot.rightFoot.body.position = pymunk.Vec2d(x+10, y)

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
                    robot.leftFoot.color = (255, int(255*(1-robot.team)), int(255*robot.team))
                    robot.rightFoot.color = (255, int(255*(1-robot.team)), int(255*robot.team))
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
                robot.leftFoot.color = (255, int(255*(1-robot.team)), int(255*robot.team))
                pos = robot.rightFoot.body.position
                pos.y = self.sideLength if ballPos.y > self.H/2 else self.H-self.sideLength
                robot.rightFoot.body.angle = math.pi / 2 if ballPos.y > self.H/2 else -math.pi / 2
                robot.rightFoot.body.position = pos
                robot.rightFoot.color = (255, int(255*(1-robot.team)), int(255*robot.team))

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

        # Moving has a small chance of falling
        if move > 0:
            r = random.random()
            if r > 0.995:
                self.fall(robot)
            robot.step(move-1)

        # Turning has a small chance of falling
        if turn > 0:
            r = random.random()
            if r > 0.995:
                self.fall(robot)
            robot.turn(turn-1)

        # Head movements have no chance of falling
        if head:
            robot.turnHead(head)

        # Kick has a higher chance of falling. Also, kick cannot be performed together with any other motion
        if kick > 0 and move == 0 and turn == 0:
            r = random.random()
            if r > 0.95:
                self.fall(robot)
            robot.kick(kick-1)

    # Get true object states
    def getFullState(self):
        return [self.ball.shape.body.position,] + [[rob.getPos(),rob.team,rob.fallen or rob.penalized] for rob in self.robots]

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
        ballDets = [isSeenInArea(self.ball.shape.body.position - pos,vec1,vec2,self.maxVisDist[0],headAngle,self.ballRadius*2)]
        robDets = [isSeenInArea(rob.getPos() - pos,vec1,vec2,self.maxVisDist[1],headAngle,Robot.totalRadius)+[robot.team == rob.team,robot.fallen or robot.penalized] for rob in self.robots if robot != rob]
        goalDets = [isSeenInArea(goal.shape.body.position - pos,vec1,vec2,self.maxVisDist[1],headAngle,self.goalPostRadius*2) for goal in self.goalposts]
        crossDets = [isSeenInArea(cross[0] - pos,vec1,vec2,self.maxVisDist[0],headAngle,self.penaltyRadius*2) for cross in self.fieldCrosses]
        lineDets = [isLineInArea(p1 - pos,p2 - pos,vec1,vec2,self.maxVisDist[1],headAngle) for p1,p2 in self.lines]
        circleDets = isSeenInArea(self.centerCircle[0] - pos,vec1,vec2,self.maxVisDist[1],headAngle,self.centerCircleRadius*2)

        # Get interactions between certain object classes
        robRobInter = [max([doesInteract(rob1[1],rob2[1],Robot.totalRadius*2) for rob1 in robDets if rob1 != rob2]) for rob2 in robDets] if self.nPlayers > 1 else [0]
        robBallInter = max([doesInteract(rob[1],ballDets[0][1],Robot.totalRadius*2) for rob in robDets])
        robPostInter = [max([doesInteract(rob[1],post[1],Robot.totalRadius*2) for rob in robDets]) for post in goalDets]
        robCrossInter = [max([doesInteract(rob[1],cross[1],Robot.totalRadius*2) for rob in robDets]) for cross in crossDets]
        ballPostInter = max([doesInteract(ballDets[0][1],post[1],self.ballRadius*8,False) for post in goalDets])
        ballCrossInter = [doesInteract(ballDets[0][1],cross[1],self.ballRadius*4,False)for cross in crossDets]

        # Random position noise and false negatives
        ballDets = [addNoise(ball, self.noiseType, max(robBallInter,ballPostInter), self.noiseMagnitude, self.randBase, self.maxVisDist[0], True) for ball in ballDets]
        robDets = [addNoise(rob, self.noiseType, robRobInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i,rob in enumerate(robDets)]
        goalDets = [addNoise(goal, self.noiseType, robPostInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i,goal in enumerate(goalDets)]
        crossDets = [addNoise(cross, self.noiseType, max(robCrossInter[i], ballCrossInter[i]), self.noiseMagnitude, self.randBase, self.maxVisDist[0], True) for i,cross in enumerate(crossDets)]
        lineDets = [addNoiseLine(line, self.noiseType, self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i,line in enumerate(lineDets)]
        circleDets = addNoise(circleDets, self.noiseType, 0, self.noiseMagnitude, self.randBase, self.maxVisDist[1])

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
                                   [SightingType.Normal,pos,Robot.totalRadius**(1-0.4*(random.random()-0.5)),random.random() > 0.5,random.random() > 0.75])
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

        if self.render and robot.id == self.visId:
            H = 540
            W = 960
            vec1.rotate(-headAngle)
            vec2.rotate(-headAngle)
            img = np.zeros((H*2,W*2,3)).astype('uint8')
            cv2.line(img,(W,H),(int(W+vec1.x*1000),int(H-vec1.y*1000)),(255,255,0))
            cv2.line(img,(W,H),(int(W+vec2.x*1000),int(H-vec2.y*1000)),(255,255,0))
            for line in lineDets:
                color = (255,255,255) if line[0] == SightingType.Normal else (127,127,127)
                cv2.line(img,(int(line[1].x+W),int(-line[1].y+H)),(int(line[2].x+W),int(-line[2].y+H)),color,self.lineWidth)
            if circleDets[0] != SightingType.NoSighting:
                color = (255,0,255) if circleDets[0] == SightingType.Normal else (127,0,127)
                cv2.circle(img,(int(circleDets[1].x+W),int(-circleDets[1].y+H)),int(circleDets[2]),color,self.lineWidth)
            for cross in crossDets:
                color = (255,255,255) if cross[0] == SightingType.Normal else (127,127,127)
                cv2.circle(img,(int(cross[1].x+W),int(-cross[1].y+H)),int(cross[2]),color,-1)
            for goal in goalDets:
                color = (255,0,0) if goal[0] == SightingType.Normal else (127,0,0)
                cv2.circle(img,(int(goal[1].x+W),int(-goal[1].y+H)),int(goal[2]),color,-1)
            for i,rob in enumerate(robDets):
                color = (0,255,0) if rob[0] == SightingType.Normal else (0,127,0)
                cv2.circle(img,(int(rob[1].x+W),int(-rob[1].y+H)),int(rob[2]),color,-1)
            for ball in ballDets:
                color = (0,0,255) if ball[0] == SightingType.Normal else (0,0,127)
                cv2.circle(img,(int(ball[1].x+W),int(-ball[1].y+H)),int(ball[2]),color,-1)

            cv2.imshow(("Robot %d" % robot.id),img)

        return ballDets,robDets,goalDets,crossDets,lineDets,circleDets
