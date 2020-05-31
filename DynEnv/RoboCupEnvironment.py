# coding=utf-8
import time

import cv2
import pygame
import pymunk.pygame_util
from gym.spaces import Tuple, MultiDiscrete, Box, MultiBinary, Dict, Space, Discrete

from .Ball import Ball
from .Goalpost import Goalpost
from .Robot import Robot
from .cutils import *
from .environment_base import EnvironmentBase, RecoDescriptor, StateSpaceDescriptor, PredictionDescriptor


class RoboCupEnvironment(EnvironmentBase):

    randomInit = False
    deterministicTurn = False
    canFall = True
    useObsRewards = True

    def __init__(self, nPlayers, render=False, observationType=ObservationType.PARTIAL, noiseType=NoiseType.REALISTIC,
                 noiseMagnitude=2, obs_space_cast=False, allowHeadTurn=False):

        super().__init__(width=1040, height=740, caption="Robot Soccer", n_players=nPlayers,
                         max_players=5, n_time_steps=5, observation_type=observationType,
                         noise_type=noiseType, render=render, obs_space_cast=obs_space_cast,
                         noise_magnitude=noiseMagnitude, max_time=12000, step_iter_cnt=50)

        # Basic settings
        self.allowHeadTurn = allowHeadTurn

        self._setup_normalization()

        self._setup_observation_space()
        self._setup_action_space()
        self._setup_reconstruction_info()

        self._setup_vision(0.4, 0.8)

        # Free kick status
        self.ballOwned = 1
        self.ballFreeCntr = 9999
        self.gracePeriod = 0

        self._init_rewards()

        self.goals = np.array([0, 0])
        self.closestID = [0, 0]

        # Bookkeeping for robots inside penalty area for illegal defender
        self.defenders = [[], []]

        # Reward settings
        self.kickDiscount = 0.5
        self.teamRewards = np.array([0.0, 0.0])
        self.robotRewards = np.array([0.0, 0.0] * self.nPlayers)
        self.penalTimes = [20000, 20000]
        self.timeDiff = 1000.0 / self.timeStep
        self.stepNum = self.maxTime / self.timeDiff / 5

        self._setup_scene()
        self._handle_collisions()

    def _setup_normalization(self):
        self.mean = 2.0 if ObservationType.PARTIAL else 1.0
        self.norm = self.mean * 2 / self.W
        self.standardNorm = 2.0 / (self.W)
        self.meanX = self.W / 2
        self.meanY = self.H / 2

    def _setup_scene(self):
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

        # Unfortunately, this needs to be done here
        self.sizeNorm = 10.0/self.penaltyRadius

        self._create_football_field()
        self._create_agents()
        self._create_ball()
        self._create_goalposts()
        self._create_penalty_spots()

    def _setup_reconstruction_info(self):
        position_xy = Box(-2, +2, shape=(2,))
        confidence = MultiBinary(1)

        # Ball
        ball_state = StateSpaceDescriptor(1, Dict({"position": position_xy,
                                                   "team": Box(-1, 1, shape=(1,)),
                                                   "confidence": confidence, }))
        #ball_pred = PredictionDescriptor(numContinuous=0, contIdx=[ ])
        ball_pred = PredictionDescriptor(numContinuous=1, contIdx=[2, ])

        # Self
        self_state = StateSpaceDescriptor(1, Dict({"position": position_xy,
                                                   "orientation": Box(-1.0, +1.0, shape=(4,)),
                                                   "active": MultiBinary(1),
                                                   "confidence": confidence, }))
        self_pred = PredictionDescriptor(numContinuous=4, numBinary=1, contIdx=[2, 3, 4, 5], binaryIdx=[7, ])

        # Robot
        robot_state = StateSpaceDescriptor(4, Dict({"position": position_xy,
                                                    "orientation": Box(-1.0, +1.0, shape=(2,)),
                                                    "team": Box(-1, 1, shape=(1,)),
                                                    "active": MultiBinary(1),
                                                    "confidence": confidence, }))

        #robot_pred = PredictionDescriptor(numContinuous=1, numBinary=1, contIdx=[4], binaryIdx=[5,])
        robot_pred = PredictionDescriptor(numContinuous=3, numBinary=1, contIdx=[2, 3, 4], binaryIdx=[5, ])


        self.recoDescriptor = RecoDescriptor(featureGridSize=(1, 1),
                                             fullStateSpace=[ball_state, robot_state],
                                             targetDefs=[ball_pred, robot_pred])

    def _init_rewards(self):
        self.episodeRewards = 0
        self.episodePosRewards = 0
        self.episodeObsRewards = 0

    def _create_penalty_spots(self):
        # Penalty spots (for penalized robots)
        self.penaltySpots = [
            [
                [[self.sideLength + (i + 1) * Robot.totalRadius * 3, self.sideLength] for i in range(7)] +
                [[self.sideLength + (i + 1) * Robot.totalRadius * 3, self.H - self.sideLength] for i in range(7)]
            ],  # team  1
            [
                [[self.W - self.sideLength - (i + 1) * Robot.totalRadius * 3, self.sideLength] for i in range(7)] +
                [[self.W - self.sideLength - (i + 1) * Robot.totalRadius * 3, self.H - self.sideLength] for i in
                 range(7)]
            ],  # team -1
        ]

    def _create_football_field(self):
        self.lines = [
            # Outer lines
            (pymunk.Vec2d(self.sideLength, self.sideLength),
             pymunk.Vec2d(self.sideLength, self.H - self.sideLength),
             1, 0),
            (pymunk.Vec2d(self.W - self.sideLength, self.sideLength),
             pymunk.Vec2d(self.W - self.sideLength, self.H - self.sideLength),
             -1, 0),
            (pymunk.Vec2d(self.sideLength, self.sideLength),
             pymunk.Vec2d(self.W - self.sideLength, self.sideLength),
             0, 1),
            (pymunk.Vec2d(self.sideLength, self.H - self.sideLength),
             pymunk.Vec2d(self.W - self.sideLength, self.H - self.sideLength),
             0, -1),

            # Middle Line
            (pymunk.Vec2d(self.W / 2, self.sideLength),
             pymunk.Vec2d(self.W / 2, self.H - self.sideLength),
             0, 0),

            # Penalty Box #1
            (pymunk.Vec2d(self.sideLength, self.H / 2 - self.penaltyWidth),
             pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 - self.penaltyWidth),
             1, 0.37),
            (pymunk.Vec2d(self.sideLength, self.H / 2 + self.penaltyWidth),
             pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 + self.penaltyWidth),
             1, -0.37),
            (pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 - self.penaltyWidth),
             pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 + self.penaltyWidth),
             0.87, 0),

            # Penalty Box #2
            (pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 - self.penaltyWidth),
             pymunk.Vec2d(self.W - self.sideLength, self.H / 2 - self.penaltyWidth),
             -1, 0.37),
            (pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 + self.penaltyWidth),
             pymunk.Vec2d(self.W - self.sideLength, self.H / 2 + self.penaltyWidth),
             -1, -0.37),
            (pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 - self.penaltyWidth),
             pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 + self.penaltyWidth),
             -0.87, 0)
        ]

        self.centerCircle = [pymunk.Vec2d((self.W // 2, self.H // 2)), self.centerCircleRadius]

        self.penaltyCrosses = [
            (pymunk.Vec2d(self.W // 2, self.H // 2), 0, 0),
            (pymunk.Vec2d(self.sideLength + self.penaltyDist, self.H // 2), 1, 0),
            (pymunk.Vec2d(self.W - (self.sideLength + self.penaltyDist), self.H // 2), -1, 0),
        ]

        self.lineCrosses = [
            # Four corners
            (pymunk.Vec2d(self.sideLength, self.sideLength), 1, 1, 0),
            (pymunk.Vec2d(self.sideLength, self.H - self.sideLength), 1, -1, 0),
            (pymunk.Vec2d(self.W - self.sideLength, self.sideLength), -1, 1, 0),
            (pymunk.Vec2d(self.W - self.sideLength, self.H - self.sideLength), -1, -1, 0),
            # Middle line
            (pymunk.Vec2d(self.W / 2, self.sideLength), 0, 1, 0),
            (pymunk.Vec2d(self.W / 2, self.H - self.sideLength), 0, -1, 0),
            # Middle Line and circle
            (pymunk.Vec2d(self.W / 2, self.H / 2 - self.centerCircleRadius * 2), 0, 0.5, 0),
            (pymunk.Vec2d(self.W / 2, self.H / 2 + self.centerCircleRadius * 2), 0, -0.5, 0),
            # Penalty Box 1
            (pymunk.Vec2d(self.sideLength, self.H / 2 - self.penaltyWidth), 1, 0.37, 0),
            (pymunk.Vec2d(self.sideLength, self.H / 2 + self.penaltyWidth), 1, -0.37, 0),
            (pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 - self.penaltyWidth), 0.87, 0.37, 0),
            (pymunk.Vec2d(self.sideLength + self.penaltyLength, self.H / 2 + self.penaltyWidth), 0.87, -0.37, 0),
            # Penalty Box 2
            (pymunk.Vec2d(self.W - self.sideLength, self.H / 2 - self.penaltyWidth), -1, 0.37, 0),
            (pymunk.Vec2d(self.W - self.sideLength, self.H / 2 + self.penaltyWidth), -1, -0.37, 0),
            (pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 - self.penaltyWidth), -0.87, 0.37, 0),
            (pymunk.Vec2d(self.W - self.sideLength - self.penaltyLength, self.H / 2 + self.penaltyWidth), -0.87, -0.37, 0),
        ]

    def _handle_collisions(self):
        # Handle robot-goalpost collision
        self._add_collision_handler(CollisionType.Robot, CollisionType.Goalpost, begin=None,
                                    post_solve=self.goalpostCollision, separate=self.separate)
        # Handle robot-robot collision
        self._add_collision_handler(CollisionType.Robot, CollisionType.Robot, self.robotPushingDet, self.robotCollision,
                                    self.separate)
        # Handle robot-ball collision
        self._add_collision_handler(CollisionType.Robot, CollisionType.Ball, begin=self.ballCollision)

    def _create_robot_spots(self):

        if self.randomInit:
            xLimits = [self.sideLength + 10, self.sideLength + 50, self.sideLength + 250, self.sideLength + 450,
                       self.sideLength + 650, self.sideLength + 850, self.sideLength + 890]
            yLimits = [self.sideLength + 20, self.sideLength + 300, self.sideLength + 580]

            robotSpots = []

            for i in range(len(xLimits)-1):
                xBeg = xLimits[i]
                xEnd = xLimits[i+1]

                ys = [yLimits[0], yLimits[-1]] if i == 0 or i == (len(xLimits) - 2) else yLimits

                for j in range(len(ys)-1):
                    yBeg = ys[j]
                    yEnd = ys[j+1]

                    x = xBeg + random.random() * (xEnd - xBeg)
                    y = yBeg + random.random() * (yEnd - yBeg)

                    robotSpots.append((x,y))

            self.robotSpots = [
                [robotSpots[0],],
                [robotSpots[-1],]
            ]
            perm = np.random.permutation(8) + 1
            for i, p in enumerate(perm):
                self.robotSpots[ 0 if i < 4 else 1] += [robotSpots[p],]

        else:
            centX = self.W / 2
            self.robotSpots = [
                # Kickoff team
                [(centX - (self.ballRadius * 2 + Robot.totalRadius) - random.random() * 50,
                  self.H / 2 + (random.random() - 0.5) * 25),
                 (centX - (Robot.totalRadius + self.lineWidth*2) - random.random() * 50,
                  self.sideLength + self.fieldH / 4 + (random.random() - 0.5) * 50),
                 (centX - (Robot.totalRadius + self.lineWidth*2) - random.random() * 50,
                  self.sideLength + 3 * self.fieldH / 4 + (random.random() - 0.5) * 50),
                 (centX - (self.fieldW / 4) - (random.random() - 0.5) * 50,
                  self.sideLength + self.fieldH / 2 + (random.random() - 0.5) * 50),
                 (self.sideLength + 20, self.H / 2 + (random.random() - 0.5) * 50)],
                # Opposing team
                [(centX + (self.centerCircleRadius * 2 + Robot.totalRadius + self.lineWidth / 2) + random.random() * 50,
                  self.H / 2 + (random.random() - 0.5) * 50),
                 (centX + (Robot.totalRadius + self.lineWidth / 2 + self.centerCircleRadius) + random.random() * 50,
                  self.sideLength + self.fieldH / 4 + (random.random() - 0.5) * 50),
                 (centX + (Robot.totalRadius + self.lineWidth / 2 + self.centerCircleRadius) + random.random() * 50,
                  self.sideLength + 3 * self.fieldH / 4 + (random.random() - 0.5) * 50),
                 (centX + (self.sideLength + self.fieldW / 4) + random.random() * 50,
                  self.sideLength + self.fieldH / 2 + (random.random() - 0.5) * 50),
                 (self.W - (self.sideLength + 20), self.H / 2 + (random.random() - 0.5) * 50)],
            ]

    def _create_goalposts(self):
        self.goalposts = [
            Goalpost(self.sideLength, self.H / 2 + self.goalWidth, 1, -0.27),
            Goalpost(self.sideLength, self.H / 2 - self.goalWidth, 1, 0.27),
            Goalpost(self.W - self.sideLength, self.H / 2 + self.goalWidth, -1, -0.27),
            Goalpost(self.W - self.sideLength, self.H / 2 - self.goalWidth, -1, 0.27),
        ]
        for goal in self.goalposts:
            self.space.add(goal.shape.body, goal.shape)

    def _create_agents(self):
        # Spot setup
        self._create_robot_spots()

        spotIds1 = np.random.permutation(self.maxPlayers)
        spotIds2 = np.random.permutation(self.maxPlayers)

        self.agents = [Robot(self.robotSpots[0][id], 1, i) for i, id in enumerate(spotIds1) if i < self.nPlayers] \
                          + [Robot(self.robotSpots[1][id], -1, self.nPlayers + i) for i, id in enumerate(spotIds2) if
                             i < self.nPlayers]

        if self.deterministicTurn:
            for rob in self.agents:
                rob.headAngle = rob.team * rob.headMaxAngle

        for robot in self.agents:
            self.space.add(robot.leftFoot.body, robot.leftFoot, robot.rightFoot.body, robot.rightFoot, robot.joint,
                           robot.rotJoint)

    def _create_ball(self):
        if self.randomInit:
            x = random.random()*self.fieldW + self.sideLength
            y = random.random()*self.fieldH + self.sideLength
            self.ballOwned = int(random.random() > 0.4)
            if self.ballOwned != 0 and random.random() > 0.5:
                self.ballOwned *= -1
        else:
            x = self.W // 2
            y = self.H // 2
        self.ball = Ball(x, y, self.ballRadius)
        self.space.add(self.ball.shape.body, self.ball.shape)

    def _setup_action_space(self):
        if self.allowHeadTurn:
            self.action_space = Tuple((MultiDiscrete([5, 3, 3]), Box(low=-3, high=3, shape=(1,))))
        else:
            self.action_space = Tuple((MultiDiscrete([5, 3, 3, 7]),))

    def _create_observation_space(self):

        # construct components
        pos_xy = Box(-self.mean * 2, +self.mean * 2, shape=(2,))
        pos_radial = Box(-1, +1, shape=(3,))
        team = Box(-1, 1, shape=(1,))

        line_space = Dict({
            "position": pos_radial,
            "type": Box(-1, +1, shape=(2,)),
            #"end1": pos_xy,
            #"end2": pos_xy,
        })
        cross_space = goalpost_space = center_circle_space = Dict({
            "position": pos_radial,
            "radius": Box(-self.mean * 2, +self.mean * 2, shape=(1,)),
            "type": Box(-1, +1, shape=(2,)),
        })
        field_cross_space = Dict({
            "position": pos_radial,
            "radius": Box(-self.mean * 2, +self.mean * 2, shape=(1,)),
            "type": Box(-1, +1, shape=(2,)),
            "angle": Box(-1, +1, shape=(2,)),
        })
        robot_space = Dict({
            "position": pos_xy,
            "radius": Box(-self.mean * 2, +self.mean * 2, shape=(1,)),
            "orientation": Box(-1, 1, shape=(2,)),
            "team": team,
            "penalized or penalized": MultiBinary(1)
        })
        self_space = Dict({
            "position": pos_xy,
            "orientation": Box(-1, 1, shape=(4,)),
            "team": team,
            "penalized or penalized": MultiBinary(1)
        })
        cheat_space = Dict({
            "position": pos_xy,
            "orientation": Box(-1, 1, shape=(2,))
        })

        if self.observationType == ObservationType.FULL:
            ball_space = Dict({
                "position": pos_xy,
                "team": team,
                "closest": MultiBinary(1),
            })

            robot_space = Dict({
                "position": pos_xy,
                "orientation": Box(-1, 1, shape=(2,)),
                "team": team,
                "penalized or penalized": MultiBinary(1)
            })

            self.observation_space = Tuple([
                Tuple([
                    ball_space,
                    robot_space]),
                Tuple([
                    self_space,]),
            ])
        elif self.observationType == ObservationType.IMAGE:
            self.observation_space = Box(0, 1, shape=(4, 480, 640))
        else:

            ball_space = Dict({
                "position": pos_xy,
                "radius": Box(-self.mean * 2, +self.mean * 2, shape=(1,)),
                "team": team,
                "closest": MultiBinary(1),
            })

            self.observation_space = Tuple([
                Tuple([
                    ball_space,
                    robot_space,
                ]),
                Tuple([
                    goalpost_space,
                    cross_space,
                    field_cross_space,
                    line_space,
                    #cheat_space
                ])
                #line_space,
                #center_circle_space
            ])

    def get_class_specific_args(self):
        return [self.allowHeadTurn]

    def get_agent_locs(self):
        return [self.getFullState(agent)[1][:, [0, 1, 2, 3, 4, 5]] for agent in self.agents]

    def get_full_obs(self):
        obs = [self.getFullState(robot) for robot in self.agents]
        obs = [([o[0], o[2]], [o[1], ], (1, 1, 1)) for o in obs]
        return obs

    # Main step function
    def step(self, actions):
        #t1 = time.clock()

        # Setup reward and state variables
        self.teamRewards = np.array([0.0, 0.0])
        self.obsRewards = np.array([0.0, 0.0] * self.nPlayers)
        self.robotRewards = np.array([0.0, 0.0] * self.nPlayers)
        self.robotPosRewards = np.array([0.0, 0.0] * self.nPlayers)
        observations = []
        finished = False

        # Run simulation for 500 ms (time for every action, except the kick)
        for i in range(self.stepIterCnt):

            # Sanity check
            actionNum = 4 #if self.allowHeadTurn else 3
            if actions.shape != (len(self.agents), actionNum):
                raise Exception("Error: There must be %d actions for every robot" % actionNum)

            # Robot loop
            for action, robot in zip(actions, self.agents):
                # Apply action as first step
                if i == 0:
                    self.processAction(action, robot)

                # Update robots
                self.tick(robot)

            # Process ball position
            isOver = self.isBallOutOfField()

            # uncomment to set finished flag on goal.
            # Otherwise env will switch to kickoff state
            # finished = isOver

            # Run simulation
            self.space.step(1 / self.timeStep)

            # Decrease game timer
            self.elapsed += 1

            # Get observations every 100 ms
            if i % 10 == 9:
                if self.observationType == ObservationType.FULL:
                    observations.append(self.get_full_obs())
                else:
                    observations.append([self.getAgentVision(robot) for robot in self.agents])

            if self.renderVar:
                self._render_internal()

        self.processSeens(observations)

        self.robotRewards[:self.nPlayers] += self.teamRewards[0]
        self.robotRewards[self.nPlayers:] += self.teamRewards[1]
        self.robotRewards += self.obsRewards
        self.episodeRewards += self.robotRewards

        # Psoitive reward for logging
        self.robotPosRewards[:self.nPlayers] += max(0.0, self.teamRewards[0])
        self.robotPosRewards[self.nPlayers:] += max(0.0, self.teamRewards[1])
        self.robotPosRewards += np.clip(self.obsRewards, a_max=None, a_min=0.0)
        self.episodePosRewards += self.robotPosRewards
        self.episodeObsRewards += self.obsRewards

        info = {'Full State': self.getFullState()}
        info['Recon States'] = [self.getFullState(robot) for robot in self.agents]

        #t2 = time.clock()
        # print((t2-t1)*1000)
        if self.elapsed >= self.maxTime:
            finished = True
            info['episode_r'] = self.episodeRewards
            info['episode_p_r'] = self.episodePosRewards
            info['episode_o_r'] = self.episodeObsRewards
            info['episode_g'] = self.goals
        #print(self.episodeRewards,self.episodePosRewards,self.episodeObsRewards)

        return observations, self.robotRewards, finished, info

    # Action handler
    def processAction(self, action, robot):

        if self.deterministicTurn:
            action[3] = -3 * robot.team

        # Get 4 action types
        if len(action) < 4:
            move, turn, kick = action
            head = 0
        else:
            move, turn, kick, head = action

        if not self.allowHeadTurn:
            head -= 3

        # Sanity check for actions
        if move not in [0, 1, 2, 3, 4]:
            raise Exception("Error: Robot movement must be categorical in the range [0-4]")
        if turn not in [0, 1, 2]:
            raise Exception("Error: Robot turn must be categorical in the range [0-2]")
        if kick not in [0, 1, 2]:
            raise Exception("Error: Robot kick must be categorical in the range [0-2]")
        if np.abs(head) > 6:
            raise Exception("Error: Head turn must be between +/-6")

        # Don't allow movement or falling unless no action is being performed
        canMove = not (robot.penalized or robot.kicking or robot.fallen)

        # Moving has a small chance of falling
        if move > 0 and canMove:
            r = random.random() if self.canFall else 0
            if r > 0.999:
                self.fall(robot, False)
                return
            robot.step(move - 1)

        # Turning has a small chance of falling
        if turn > 0 and canMove:
            r = random.random() if self.canFall else 0
            if r > 0.999:
                self.fall(robot, False)
                return
            robot.turn(turn - 1)

        # Head movements have no chance of falling
        if head:
            robot.turnHead(head)

        # Kick has a higher chance of falling. Also, kick cannot be performed together with any other motion
        if kick > 0 and move == 0 and turn == 0 and canMove:
            r = random.random() if self.canFall else 0
            if r > 0.99:
                self.fall(robot, False)
                return
            robot.kick(kick - 1)

    # Drawing
    def drawStaticObjects(self):

        # Field green
        self.screen.fill((0, 255, 0))

        for line in self.lines:
            pygame.draw.line(self.screen, (255, 255, 255), line[0], line[1], self.lineWidth)

        for cross in self.penaltyCrosses:
            pygame.draw.circle(self.screen, (255, 255, 255), cross[0], self.penaltyRadius * 2, 0)

        pygame.draw.circle(self.screen, (255, 255, 255), self.centerCircle[0], self.centerCircle[1] * 2, self.lineWidth)

        self.space.debug_draw(self.draw_options)

    # Ball free kick
    def ballFreeKickProcess(self, team):
        if team == 0:
            time = 1000 / self.timeStep
            if self.gracePeriod > 0:
                self.gracePeriod -= time
                if self.gracePeriod < 0:
                    # print("Grace period over")
                    self.gracePeriod = 0
                    self.ballFreeCntr = 9999
            elif self.ballFreeCntr > 0:
                self.ballFreeCntr -= time
                if self.ballFreeCntr < 0:
                    # print("Ball Free")
                    self.ballFreeCntr = 0
                    self.ballOwned = 0
        else:
            # print("Free kick", team)
            self.ballOwned = team
            self.gracePeriod = 14999
            self.ballFreeCntr = 0

    # Detect ball movements
    def isBallOutOfField(self):

        # Setup basic parameters
        finished = False
        pos = self.ball.shape.body.position

        # Current tem rewards
        currReward = [0, 0]

        # Field edges
        outMin = self.sideLength - self.ballRadius
        outMaxX = self.W - self.sideLength + self.ballRadius
        outMaxY = self.H - self.sideLength + self.ballRadius

        # Team to award free kick
        team = 0

        # If ball is out
        if pos.y < outMin or pos.x < outMin or pos.y > outMaxY or pos.x > outMaxX:
            x = self.W / 2
            y = self.H / 2

            # If out on the sides
            team = self.agents[self.ball.lastKicked[0]].team if len(self.ball.lastKicked) else 1
            if pos.y < outMin or pos.y > outMaxY:
                x = pos.x + 50 if team < 0 else pos.x - 50
                if pos.y < outMin:
                    y = outMin + self.ballRadius
                else:
                    y = outMaxY - self.ballRadius
            # If out on the ends
            else:
                # If goal
                if pos.y < self.H / 2 + self.goalWidth and pos.y > self.H / 2 - self.goalWidth:
                    finished = True
                    if pos.x < outMin:
                        currReward[0] += -25
                        currReward[1] += 25
                        self.goals[1] += 1
                    else:
                        currReward[0] += 25
                        currReward[1] += -25
                        self.goals[0] += 1
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
                            y = self.sideLength if pos.y < self.H / 2 else self.H - self.sideLength
                    else:
                        # Kick out
                        if team > 0:
                            x = self.W - (self.sideLength + self.penaltyLength)
                        # Corner
                        else:
                            x = self.W - self.sideLength
                            y = self.sideLength if pos.y < self.H / 2 else self.H - self.sideLength

            # Move ball to middle and stop it
            self.ball.shape.body.position = pymunk.Vec2d(x, y)
            self.ball.shape.body.velocity = pymunk.Vec2d(0.0, 0.0)
            self.ball.shape.body.angular_velocity = 0.0

        # Update free kick status
        self.ballFreeKickProcess(-team)

        # Add ball movement to the reward
        if not finished:
            currReward[0] += (self.ball.shape.body.position.x - self.ball.prevPos.x) / 20
            currReward[1] -= (self.ball.shape.body.position.x - self.ball.prevPos.x) / 20

        # Update previous position
        self.ball.prevPos = self.ball.shape.body.position

        # Create discounted personal rewards for the robots involved
        for i, id in enumerate(self.ball.lastKicked):
            rew = currReward[0] * (self.kickDiscount ** i) if id < self.nPlayers else currReward[1] * (
                    self.kickDiscount ** i)
            self.robotRewards[id] += rew
            self.robotPosRewards[id] += max(0.0, rew)

        # Create personal rewards for nearby robots not touching the ball, but only negative rewards
        for robot in self.agents:
            cond1 = robot.id in self.closestID
            cond2 = (robot.getPos() - pos).length < 150
            if cond1 or cond2:
                if cond2 and not robot.penalized and self.ballOwned != robot.team and self.ballFreeCntr > 0:
                    # print("Illegal position", robot.id, robot.team)
                    if False:  # Disable this for now
                        self.penalize(robot)
                if robot.id not in self.ball.lastKicked:
                    self.robotRewards[int(robot.id)] += min(0, currReward[
                                                                   0] * self.kickDiscount if robot.id < self.nPlayers else
                    currReward[1] * self.kickDiscount)

        # Update team rewards
        self.teamRewards[0] += currReward[0] * 0.1
        self.teamRewards[1] += currReward[1] * 0.1

        # Get closest robot from both teams
        pos = self.ball.getPos()
        self.closestID[0] = np.argmin([(pos - rob.getPos()).get_length_sqrd() for rob in self.agents if rob.team > 0])
        self.closestID[1] = self.nPlayers + np.argmin(
            [(pos - rob.getPos()).get_length_sqrd() for rob in self.agents if rob.team < 0])

        return finished

    # Robot falling
    def fall(self, robot, punish=True):
        # print("Fall", robot.fallCntr, robot.team)

        # Get robot position
        pos = robot.getPos()

        # Find objects nearby
        filter = pymunk.shape_filter.ShapeFilter(categories=0b101)
        shapes = self.space.point_query(pos, 40, filter)

        # Punish robots for falling
        if punish:
            self.robotRewards[robot.id] -= 2

        # For closeby objects (that are not the robot)
        for query in shapes:
            if query.shape != robot.leftFoot and query.shape != robot.rightFoot:

                # Compute force of the robot's fall
                force = robot.velocity * robot.leftFoot.body.mass * query.shape.body.mass / 50.0

                # Get direction of the force
                dp = pos - query.shape.body.position

                # Force is proportional to the distance
                dp = -dp * force / dp.length

                # Apply force
                query.shape.body.apply_force_at_world_point(dp, pos)

                # If the fallen robots touched the ball, update the last touched variable
                if query.shape == self.ball.shape:
                    if len(self.ball.lastKicked) and robot.id not in self.ball.lastKicked:
                        self.ball.lastKicked = [robot.id] + self.ball.lastKicked
                    if len(self.ball.lastKicked) > 4:
                        self.ball.lastKicked = self.ball.lastKicked[:4]
                    if self.ballOwned != 0:
                        # print("Ball Free")
                        self.gracePeriod = 0
                        self.ballFreeCntr = 0
                        self.ballOwned = 0

        # Update the color of the fallen robot
        robot.leftFoot.color = (255, int(75 * (1 + robot.team)), int(75 * (1 - robot.team)))
        robot.rightFoot.color = (255, int(75 * (1 + robot.team)), int(75 * (1 - robot.team)))

        # Set variables
        robot.fallen = True

        # Set number of falls and getup time
        robot.fallCntr += 1
        robot.fallTime = 4000

        # If the robot fell 3 times, penalize
        if robot.fallCntr > 2:
            # print("Fallen robot", robot.fallCntr, robot.team)
            self.penalize(robot)

    def getFreePenaltySpot(self, robot):

        # Get speantly spots for team
        availableSpots = self.penaltySpots[0][0] if robot.team > 0 else self.penaltySpots[1][0]

        # Narrow dow spots based on ball location
        y = self.ball.getPos().y
        angle = -math.pi / 2 if y < self.H / 2 else math.pi / 2
        availableSpots = availableSpots[:7] if y > self.H / 2 else availableSpots[7:]

        # Select first free spot
        selectedSpot = availableSpots[0]
        for spot in availableSpots:
            available = True

            # Check if any robots are too close
            for rob in self.agents:
                if rob == robot:
                    continue
                dist = (spot - rob.getPos()).length
                if dist < Robot.totalRadius * 3:
                    available = False
                    break
            # Select first free spot
            if available:
                selectedSpot = spot
                break

        return selectedSpot, angle

    # Penlize robot
    def penalize(self, robot):

        # Update penalized status
        robot.penalized = True
        teamIdx = 0 if robot.team > 0 else 1
        robot.penalTime = self.penalTimes[teamIdx]

        # Punish the robot
        self.robotRewards[robot.id] -= self.penalTimes[teamIdx] / 2000

        # Increase penalty time for team
        self.penalTimes[teamIdx] += 10000

        # Compute robot position
        pos, angle = self.getFreePenaltySpot(robot)

        # Move feet
        robot.leftFoot.body.position = pos
        robot.leftFoot.body.angle = angle
        robot.rightFoot.body.position = pos
        robot.rightFoot.body.angle = angle

        # Stop feet, and change color
        robot.leftFoot.body.velocity = pymunk.Vec2d(0.0, 0.0)
        robot.leftFoot.body.angular_velocity = 0.0
        robot.leftFoot.color = (255, 0, 0)
        robot.rightFoot.body.velocity = pymunk.Vec2d(0.0, 0.0)
        robot.rightFoot.body.angular_velocity = 0.0
        robot.rightFoot.color = (255, 0, 0)

        # Set moving variables
        if robot.kicking and robot.jointRemoved:
            robot.kicking = False
            # If the robot was kicking, the joint between its legs was removed. It needs to be added back
            self.space.add(robot.joint)
            robot.jointRemoved = False

    # Robot update function
    def tick(self, robot):

        # Get timestep
        time = 1000 / self.timeStep

        # If the robot is moving
        if robot.moveTime > 0:
            robot.moveTime -= time

            # Move head
            if robot.headMoving != 0:
                robot.headAngle += robot.headMoving
                robot.headAngle = max(-robot.headMaxAngle, min(robot.headMaxAngle, robot.headAngle))

            # Update kicking
            if robot.kicking:
                # Get which foot
                foot = robot.rightFoot if robot.foot else robot.leftFoot

                # 500 ms into the kick the actual movement starts
                if robot.moveTime + time > 500 and robot.moveTime <= 500:
                    # Remove joint between legs to allow the leg to move independently
                    if not robot.jointRemoved:
                        self.space.remove(robot.joint)
                        robot.jointRemoved = True

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
                    foot.body.velocity = pymunk.Vec2d(0, 0)
                    robot.kicking = False
                    foot.body.position = robot.initPos

                    # Add joint back
                    if robot.jointRemoved:
                        self.space.add(robot.joint)
                        robot.jointRemoved = False

            # If the movement is over
            if robot.moveTime <= 0:
                # Stop the robot
                robot.moveTime = 0
                robot.headMoving = 0
                robot.leftFoot.body.velocity = pymunk.Vec2d(0, 0)
                robot.leftFoot.body.angular_velocity = 0.0
                robot.rightFoot.body.velocity = pymunk.Vec2d(0, 0)
                robot.rightFoot.body.angular_velocity = 0.0

        # If the robot fell
        if robot.fallen:

            robot.fallTime -= time

            if robot.fallTime < 0:

                # Is might fall again
                r = random.random()
                if r > 0.9 and not robot.penalized and self.canFall:
                    self.fall(robot, False)
                    return

                # print("Getup", robot.team)

                # Reset color and variables
                robot.leftFoot.color = (255, int(127 * (1 - robot.team)), int(127 * (1 + robot.team)))
                robot.rightFoot.color = (255, int(127 * (1 - robot.team)), int(127 * (1 + robot.team)))
                robot.fallen = False
                robot.fallCntr = 0

        # Handle penalties
        if robot.penalized:
            robot.penalTime -= time

            # If expired
            if robot.penalTime <= 0:
                # print("Unpenalized")

                # Reset all variables
                robot.penalTime = 0
                robot.penalized = False
                robot.fallCntr = 0
                robot.fallen = False

                # Get robot position
                pos, angle = self.getFreePenaltySpot(robot)

                # Move feet
                robot.leftFoot.body.position = pos
                robot.leftFoot.body.angle = angle
                robot.leftFoot.color = (255, int(127 * (1 - robot.team)), int(127 * (1 + robot.team)))
                robot.rightFoot.body.position = pos
                robot.rightFoot.body.angle = angle
                robot.rightFoot.color = (255, int(127 * (1 - robot.team)), int(127 * (1 + robot.team)))
        else:
            teamIdx = 0 if robot.team > 0 else 1

            # Get robot and penalty positions
            pos = robot.getPos()
            robX = self.W - pos.x if teamIdx else pos.x
            penX = self.sideLength + self.penaltyLength + self.lineWidth / 2

            # If robot is in the penalty box
            if robX < penX and pos.y > (self.H / 2 - self.penaltyWidth) and pos.y < (self.H / 2 + self.penaltyWidth):

                # If not in the defenders, add it or penalize if limit is reached
                if robot.id not in self.defenders[teamIdx]:
                    if len(self.defenders[teamIdx]) >= 2:
                        # print("Illegal defender")
                        self.penalize(robot)
                    else:
                        self.defenders[teamIdx].append(robot.id)

            # If the robot is not in the box anymore, remove it from the defender list
            elif robot.id in self.defenders[teamIdx]:
                self.defenders[teamIdx].remove(robot.id)

        # If robot is about to leave the field, penalize
        pos = robot.getPos()
        if pos.y < 0 or pos.x < 0 or pos.y > self.H or pos.x > self.W:
            self.penalize(robot)

        # If robot got closer to the ball, reward
        if pos != robot.prevPos:
            # Reward only closes rotobs from each team
            if (robot.id in self.closestID) and not robot.penalized:
                ballPos = self.ball.getPos()
                diff = (pos - ballPos).length - (robot.prevPos - ballPos).length

                self.robotRewards[self.agents.index(robot)] -= diff * 0.05
                self.robotPosRewards[self.agents.index(robot)] += max(0.0, -diff * 0.05)

            robot.prevPos = pos

    # Called when robots begin touching
    def robotPushingDet(self, arbiter, space, data):

        # Get objects involved
        robot1 = next(robot for robot in self.agents if
                      (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        robot2 = next(robot for robot in self.agents if
                      (robot.leftFoot == arbiter.shapes[1] or robot.rightFoot == arbiter.shapes[1]))

        # Get velocities and positions
        v1 = arbiter.shapes[0].body.velocity
        v2 = arbiter.shapes[1].body.velocity
        p1 = robot1.getPos()
        p2 = robot2.getPos()
        dp = p1 - p2

        # Robot might be pushing if it's walking towards the other
        robot1.mightPush = v1.length > 1 and math.cos(angle(dp) - angle(v1)) < -0.4
        robot2.mightPush = v2.length > 1 and math.cos(angle(dp) - angle(v2)) > 0.4

        # Set touching and touch counter variables
        robot1.touching = True
        robot2.touching = True
        robot1.touchCntr = 0
        robot2.touchCntr = 0

        return True

    # Called when robots are touching (before collision is computed)
    def robotCollision(self, arbiter, space, data):

        if not self.canFall:
            return

        # Get objects involved
        robot1 = next(robot for robot in self.agents if
                      (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))
        robot2 = next(robot for robot in self.agents if
                      (robot.leftFoot == arbiter.shapes[1] or robot.rightFoot == arbiter.shapes[1]))

        # The two legs might collide
        if robot1 == robot2:
            return

        # Increment touching
        if not (robot1.fallen or robot1.penalized):
            robot1.touchCntr += 1
        if not (robot2.fallen or robot2.penalized):
            robot2.touchCntr += 1

        # Compute fall probability thresholds - the longer the robots are touching the more likely they will fall
        normalThresh = 0.9999
        pushingThresh = 0.99995

        # Determine if robots fall
        r = random.random()
        if r > (pushingThresh if robot1.mightPush else normalThresh) ** robot1.touchCntr and not robot1.fallen:
            self.fall(robot1, robot1.mightPush)
            robot1.touchCntr = 0
        r = random.random()
        if r > (pushingThresh if robot2.mightPush else normalThresh) ** robot2.touchCntr and not robot2.fallen:
            self.fall(robot2, robot2.mightPush)
            robot2.touchCntr = 0

        # Penalize robots for pushing
        if robot1.mightPush and not robot2.mightPush and robot2.fallen and robot1.team != robot2.team:
            # print("Robot 1 Pushing")
            self.penalize(robot1)
            robot1.touchCntr = 0
        elif robot2.mightPush and not robot1.mightPush and robot1.fallen and robot1.team != robot2.team:
            # print("Robot 2 Pushing")
            self.penalize(robot2)
            robot2.touchCntr = 0

    # Called when robots stop touching
    def separate(self, arbiter, space, data):

        # Get robot
        robots = [robot for robot in self.agents if
                  (robot.leftFoot in arbiter.shapes or robot.rightFoot in arbiter.shapes)]

        # Reset collision variables
        for robot in robots:
            robot.touching = False
            robot.mightPush = False
            robot.touchCntr = 0

    # Called when robot collides with goalpost
    def goalpostCollision(self, arbiter, space, data):

        if not self.canFall:
            return

        # Get robot
        robot = next(robot for robot in self.agents if
                     (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))

        # Don't make a fallen robot fall again from touching the post
        if robot.fallen:
            robot.touchCntr = 0
            return

        # Set things on first run
        if not robot.touching:
            robot.touching = True
            robot.touchCntr = 0

        # Increment touch counter and compute fall probability threshold
        robot.touchCntr += 1
        pushingThresh = 0.9998 ** robot.touchCntr

        # Determine if robot falls
        r = random.random()
        if r > pushingThresh:
            self.fall(robot)

    # Called when robot touches the ball
    def ballCollision(self, arbiter, space, data):

        # Get robot
        robot = next(robot for robot in self.agents if
                     (robot.leftFoot == arbiter.shapes[0] or robot.rightFoot == arbiter.shapes[0]))

        if self.ballOwned != 0:
            if robot.team != self.ballOwned and not robot.penalized and self.canFall:
                self.penalize(robot)
            else:
                # print("Ball Free")
                self.ballOwned = 0
                self.gracePeriod = 0
                self.ballFreeCntr = 0

        # Shift lastKicked array
        self.ball.lastKicked = [robot.id] + self.ball.lastKicked
        if len(self.ball.lastKicked) > 4:
            self.ball.lastKicked = self.ball.lastKicked[:4]

        return True

    # Get true object state for a robot
    def getFullState(self, agent=None):

        if agent is None:
            state = [np.array([[normalize(rob.getPos()[0], self.standardNorm, 0),
                                normalize(rob.getPos()[1], self.standardNorm, 0),
                                math.cos(rob.getAngle()), math.sin(rob.getAngle()),
                                rob.team,
                                int(rob.fallen or rob.penalized)]
                               for rob in self.agents]).astype('float32'),

                     np.array([normalize(self.ball.getPos()[0], self.standardNorm, 0),
                               normalize(self.ball.getPos()[1], self.standardNorm, 0),
                               self.ballOwned]).astype('float32')]
        else:
            # flip axes for team -1
            team = agent.team
            pos = agent.getPos()

            state = [
                np.array([
                    [normalizeAfterScale(self.ball.getPos()[0], self.standardNorm, self.meanX, team),
                     normalizeAfterScale(self.ball.getPos()[1], self.standardNorm, self.meanY, team),
                     self.ballOwned * agent.team,
                     agent.id in self.closestID], ]).astype('float32'),

                np.array([[
                    normalizeAfterScale(agent.getPos()[0], self.standardNorm, self.meanX, team),
                    normalizeAfterScale(agent.getPos()[1], self.standardNorm, self.meanY, team),
                    math.cos(agent.getAngle(team) + agent.headAngle), math.sin(agent.getAngle(team) + agent.headAngle),
                    math.cos(agent.headAngle), math.sin(agent.headAngle),
                    agent.team,
                    int(agent.fallen or agent.penalized)], ]).astype('float32'),

                np.array([
                    [normalizeAfterScale(rob.getPos()[0], self.standardNorm, self.meanX, team),
                     normalizeAfterScale(rob.getPos()[1], self.standardNorm, self.meanY, team),
                     math.cos(rob.getAngle(team)), math.sin(rob.getAngle(team)),
                     rob.team * agent.team,
                     int(rob.fallen or rob.penalized)]
                    for rob in self.agents if rob != agent]).astype('float32')]
        return state

    # Getting vision
    def getAgentVision(self, agent):

        # Get position and orientation
        pos = agent.getPos()
        angle = agent.getAngle(agent.team)
        headAngle = angle + agent.headAngle

        # FoV
        angle1 = headAngle + agent.fieldOfView
        angle2 = headAngle - agent.fieldOfView

        # Edge of field of view
        vec1 = pymunk.Vec2d(1, 0)
        vec1.rotate(angle1)

        # Other edge of field of view
        vec2 = pymunk.Vec2d(1, 0)
        vec2.rotate(angle2)

        # Check if objects are seen
        ballDets = [isSeenInArea(self.ball.shape.body.position - pos, vec1, vec2, self.maxVisDist[0], headAngle,
                                 self.ballRadius * 2) + [self.ballOwned * agent.team]]
        robDets = [isSeenInArea(rob.getPos() - pos, vec1, vec2, self.maxVisDist[1], headAngle, Robot.totalRadius) + [
            rob.getAngle() - headAngle, agent.team * rob.team, agent.fallen or agent.penalized] for rob in self.agents
                   if agent != rob]
        goalDets = [isSeenInArea(goal.shape.body.position - pos, vec1, vec2, self.maxVisDist[1], headAngle,
                                 goal.radius) + [goal.side, goal.dir] for goal in self.goalposts]
        crossDets = [isSeenInArea(cross[0] - pos, vec1, vec2, self.maxVisDist[0], headAngle, self.penaltyRadius) + [cross[1], cross[2]] for
                     cross in self.penaltyCrosses]
        fieldCrossDets = [isSeenInArea(cross[0] - pos, vec1, vec2, self.maxVisDist[0], headAngle, self.penaltyRadius) + [cross[1], cross[2], cross[3]-headAngle] for
                     cross in self.lineCrosses]
        lineDets = [isLineInArea(p1 - pos, p2 - pos, vec1, vec2, self.maxVisDist[1], headAngle) + [tx, ty] for p1, p2, tx, ty in
                    self.lines]
        circleDets = isSeenInArea(self.centerCircle[0] - pos, vec1, vec2, self.maxVisDist[1], headAngle,
                                  self.centerCircleRadius * 2, False)

        # Get interactions between certain object classes
        robRobInter = [max([doesInteract(rob1[1], rob2[1], Robot.totalRadius * 2) for rob1 in robDets if rob1 != rob2])
                       for rob2 in robDets] if self.nPlayers > 1 else [0]
        robBallInter = max([doesInteract(rob[1], ballDets[0][1], Robot.totalRadius * 2) for rob in robDets])
        robPostInter = [max([doesInteract(rob[1], post[1], Robot.totalRadius * 2) for rob in robDets]) for post in
                        goalDets]
        robCrossInter = [max([doesInteract(rob[1], cross[1], Robot.totalRadius * 2) for rob in robDets]) for cross in
                         crossDets]
        robFCrossInter = [max([doesInteract(rob[1], cross[1], Robot.totalRadius * 2) for rob in robDets]) for cross in
                         fieldCrossDets]
        ballPostInter = max([doesInteract(ballDets[0][1], post[1], self.ballRadius * 8, False) for post in goalDets])
        ballCrossInter = [doesInteract(ballDets[0][1], cross[1], self.ballRadius * 4, False) for cross in crossDets]

        # Random position noise and false negatives
        [addNoise(ball, self.noiseType, max(robBallInter, ballPostInter), self.noiseMagnitude, self.randBase,
                  self.maxVisDist[0], True) for ball in ballDets]
        [addNoise(rob, self.noiseType, robRobInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for
         i, rob in enumerate(robDets)]
        [addNoise(goal, self.noiseType, robPostInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for
         i, goal in enumerate(goalDets)]
        [addNoise(cross, self.noiseType, max(robCrossInter[i], ballCrossInter[i]), self.noiseMagnitude, self.randBase,
                  self.maxVisDist[0], True) for i, cross in enumerate(crossDets)]
        [addNoise(cross, self.noiseType, robFCrossInter[i], self.noiseMagnitude, self.randBase,
                  self.maxVisDist[0], angleNoise = True) for i, cross in enumerate(fieldCrossDets)]
        addNoise(circleDets, self.noiseType, 0, self.noiseMagnitude, self.randBase, self.maxVisDist[1])
        [addNoiseLine(line, self.noiseType, self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i, line in
         enumerate(lineDets)]

        robotsSeen = np.array([rob[0] != SightingType.NoSighting for rob in robDets]).astype('uint8')
        ballsSeen = (ballDets[0][0] != SightingType.NoSighting and ballDets[0][0] != SightingType.Misclassified)

        # Balls and crosses might by miscalssified - move them in the other list
        for ball in ballDets:
            if ball[0] == SightingType.Misclassified:
                crossDets.append([SightingType.Normal, ball[1], ball[2], random.randint(-1,1), random.randint(-1,1)])
        for cross in crossDets:
            if cross[0] == SightingType.Misclassified:
                ballDets.append([SightingType.Normal, cross[1], cross[2], 0])

        # Remove occlusion and misclassified originals
        ballDets = [ball for i, ball in enumerate(ballDets) if
                    ball[0] != SightingType.NoSighting and ball[0] != SightingType.Misclassified]
        robDets = [rob for i, rob in enumerate(robDets) if rob[0] != SightingType.NoSighting]
        goalDets = [goal for i, goal in enumerate(goalDets) if goal[0] != SightingType.NoSighting]
        crossDets = [cross for i, cross in enumerate(crossDets) if
                     cross[0] != SightingType.NoSighting and cross[0] != SightingType.Misclassified]
        fieldCrossDets = [cross for i, cross in enumerate(fieldCrossDets) if
                          cross[0] != SightingType.NoSighting and cross[0] != SightingType.Misclassified]
        lineDets = [line for i, line in enumerate(lineDets) if line[0] != SightingType.NoSighting]

        numLandMarks = len(fieldCrossDets) + len(lineDets) + len(crossDets) + len(goalDets)

        # Random false positives
        for i in range(10):
            if random.random() < self.randBase:
                c = random.randint(0, 5)
                d = random.random() * math.sqrt(self.maxVisDist[1])
                a = random.random() * 2 * agent.fieldOfView - agent.fieldOfView
                pos = pymunk.Vec2d(d, 0)
                pos.rotate(a)
                if c == 0:
                    ballDets.insert(len(ballDets),
                                    [SightingType.Normal, pos,
                                     self.ballRadius * 2 * (1 - 0.4 * (random.random() - 0.5)), 0])
                elif c == 1:
                    robDets.insert(len(robDets),
                                   [SightingType.Normal, pos, Robot.totalRadius * (1 - 0.4 * (random.random() - 0.5)),
                                    (random.random() - 0.5) * 2 * math.pi,
                                    (-1) ** int(random.random() > 0.5), random.random() > 0.9])
                elif c == 2:
                    goalDets.insert(len(goalDets),
                                    [SightingType.Normal, pos,
                                     self.goalPostRadius * (1 - 0.4 * (random.random() - 0.5)), random.randint(-1,1), random.randint(-1,1)])
                elif c == 3:
                    crossDets.insert(len(crossDets),
                                     [SightingType.Normal, pos,
                                      self.penaltyRadius * (1 - 0.4 * (random.random() - 0.5)), random.randint(-1,1), random.randint(-1,1)])
                elif c == 4:
                    fieldCrossDets.insert(len(crossDets),
                                     [SightingType.Normal, pos,
                                      self.penaltyRadius * (1 - 0.4 * (random.random() - 0.5)), random.randint(-1,1), random.randint(-1,1), random.random()*math.pi*2])

        # FP Balls near robots
        if self.noiseType == NoiseType.REALISTIC:
            for rob in robDets:
                if rob[0] == SightingType.Normal and random.random() < self.randBase * 10 and rob[1].length < 250:
                    if random.random() < self.randBase * 8:
                        rob[0] = SightingType.NoSighting
                    offset = pymunk.Vec2d(2 * random.random() - 1.0, 2 * random.random() - 1.0) * Robot.totalRadius
                    ballDets.insert(len(ballDets),
                                    [SightingType.Normal, rob[1] + offset,
                                     self.ballRadius * 2 * (1 - 0.4 * (random.random() - 0.5)), 0])

        if self.observationType == ObservationType.IMAGE:

            # Initialize images
            bottomCamImg = np.zeros((4, 480, 640))
            topCamImg = np.zeros((4, 480, 640))

            for line in lineDets:
                # Points to transform: [start, start+thickness, end]
                linevec = np.array([
                    [(-line[1].y - line[2].y) / 2, 0, (line[1].x + line[2].x) / 2, 1],
                    [(-line[1].y - line[2].y) / 2 + self.lineWidth / 2, 0, (line[1].x + line[2].x) / 2, 1],
                    [-line[1].y, 0, line[1].x, 1],
                    [-line[2].y, 0, line[2].x, 1]
                ]).transpose()

                # Project points and estimate radius (projected size of line thickness)
                tProj, tRad, bProj, bRad = projectPoints(linevec)

                # Draw
                cv2.line(topCamImg[3], (int(tProj[0, 2]), int(tProj[1, 2])), (int(tProj[0, 3]), int(tProj[1, 3])), 1,
                         tRad)
                cv2.line(bottomCamImg[3], (int(bProj[0, 2]), int(bProj[1, 2])), (int(bProj[0, 3]), int(bProj[1, 3])), 1,
                         bRad)

            if circleDets[0] != SightingType.NoSighting:

                # Rotated directional vector
                ellipseOffs = pymunk.Vec2d(circleDets[2], 0)
                ellipseOffs.rotate(math.pi / 4)

                # Points to transform: [center, 6 more points on the circle]
                circlevec = np.array([
                    [-circleDets[1].y, 0, circleDets[1].x, 1],
                    [-circleDets[1].y, 0, circleDets[1].x - circleDets[2], 1],
                    [-circleDets[1].y, 0, circleDets[1].x + circleDets[2], 1],
                    [-circleDets[1].y - circleDets[2], 0, circleDets[1].x, 1],
                    [-circleDets[1].y + circleDets[2], 0, circleDets[1].x, 1],
                    [-circleDets[1].y - ellipseOffs.y, 0, circleDets[1].x + ellipseOffs.x, 1],
                    [-circleDets[1].y + ellipseOffs.x, 0, circleDets[1].x - ellipseOffs.y, 1],
                    [-circleDets[1].y + ellipseOffs.y, 0, circleDets[1].x + ellipseOffs.x, 1],
                    [-circleDets[1].y - ellipseOffs.x, 0, circleDets[1].x - ellipseOffs.y, 1],
                ]).transpose()

                # Project points and estimate radius (projected size of line thickness)
                tProj, tRad, bProj, bRad = projectPoints(circlevec, False)

                # estimate line thickness from center distance
                tThickness = 15 - max(0, min(14, int(circleDets[1].length / 40)))
                bThickness = 30 - max(0, min(29, int(circleDets[1].length / 20)))

                # Estimate conic parameters
                tParams = estimateConic(tProj[:, 1:] - tProj[:, 0:1])
                bParams = estimateConic(bProj[:, 1:] - bProj[:, 0:1])

                # Get [x,y] coordinates of the conic for y in [0,480)
                # x1 and x2 are the two curves that make up the conic (they might be separate due to field of vision)
                tx1, tx2 = getConicPoints(480, tProj[:, 0], tParams)
                bx1, bx2 = getConicPoints(480, bProj[:, 0], bParams)

                # Draw polygon on points
                cv2.polylines(topCamImg[3], [tx1], False, 1, tThickness)
                cv2.polylines(topCamImg[3], [tx2], False, 1, tThickness)
                cv2.polylines(bottomCamImg[3], [bx1], False, 1, bThickness)
                cv2.polylines(bottomCamImg[3], [bx2], False, 1, bThickness)

                # Connect the first and last elements of the two curves, unless they are at the edges of the images
                if tx1.shape[0]:
                    if tx1[0, 1]:
                        cv2.line(topCamImg[3], tuple(tx1[0]), tuple(tx2[0]), 1, tThickness)
                    if tx1[-1, 1] < 480 - 1:
                        cv2.line(topCamImg[3], tuple(tx1[-1]), tuple(tx2[-1]), 1, tThickness)
                if bx1.shape[0]:
                    if bx1[0, 1]:
                        cv2.line(bottomCamImg[3], tuple(bx1[0]), tuple(bx2[0]), 1, bThickness)
                    if bx1[-1, 1] < 480 - 1:
                        cv2.line(bottomCamImg[3], tuple(bx1[-1]), tuple(bx2[-1]), 1, bThickness)

            for rob in robDets:
                # Points to transform: [bottom left, bottom right, top left, top right]
                robvec = np.array([
                    [-rob[1].y - rob[2], 0, rob[1].x, 1],
                    [-rob[1].y + rob[2], 58, rob[1].x, 1]
                ]).transpose()

                # Project points (without radius estimation)
                tProj, tRad, bProj, bRad = projectPoints(robvec, False)

                # Draw
                cv2.rectangle(topCamImg[1], (int(tProj[0, 0]), int(tProj[1, 0])), (int(tProj[0, 1]), int(tProj[1, 1])),
                              1, -1)
                cv2.rectangle(bottomCamImg[1], (int(bProj[0, 0]), int(bProj[1, 0])),
                              (int(bProj[0, 1]), int(bProj[1, 1])), 1, -1)

            for goal in goalDets:
                # Points to transform: [bottom, bottom+thickness, top]
                goalvec = np.array([
                    [-goal[1].y, 0, goal[1].x, 1],
                    [-goal[1].y + goal[2] / 2, 0, goal[1].x, 1],
                    [-goal[1].y, 80, goal[1].x, 1]
                ]).transpose()

                # Project points and estimate radius (projected size of goal thickness)
                tProj, tRad, bProj, bRad = projectPoints(goalvec)

                # Draw
                cv2.line(topCamImg[2], (int(tProj[0, 0]), int(tProj[1, 0])), (int(tProj[0, 2]), int(tProj[1, 2])), 1,
                         tRad)
                cv2.line(bottomCamImg[2], (int(bProj[0, 0]), int(bProj[1, 0])), (int(bProj[0, 2]), int(bProj[1, 2])), 1,
                         bRad)

            for cross in crossDets:
                # Points to transform: [center, center+thickness]
                crossvec = np.array([
                    [-cross[1].y, 0, cross[1].x, 1],
                    [-cross[1].y + cross[2] / 2, 0, cross[1].x, 1]
                ]).transpose()

                # Project points and estimate radius (projected size of cross radius)
                tProj, tRad, bProj, bRad = projectPoints(crossvec)

                # Draw
                cv2.circle(topCamImg[3], (int(tProj[0, 0]), int(tProj[1, 0])), tRad, 1, -1)
                cv2.circle(bottomCamImg[3], (int(bProj[0, 0]), int(bProj[1, 0])), bRad, 1, -1)

            for cross in fieldCrossDets:
                # Points to transform: [center, center+thickness]
                crossvec = np.array([
                    [-cross[1].y, 0, cross[1].x, 1],
                    [-cross[1].y + cross[2] / 2, 0, cross[1].x, 1]
                ]).transpose()

                # Project points and estimate radius (projected size of cross radius)
                tProj, tRad, bProj, bRad = projectPoints(crossvec)

                # Draw
                cv2.circle(topCamImg[3], (int(tProj[0, 0]), int(tProj[1, 0])), tRad, 1, -1)
                cv2.circle(bottomCamImg[3], (int(bProj[0, 0]), int(bProj[1, 0])), bRad, 1, -1)

            for ball in ballDets:
                # Points to transform: [center, center+thickness]
                ballvec = np.array([
                    [-ball[1].y, ball[2] / 2, ball[1].x, 1],
                    [-ball[1].y + ball[2] / 2, ball[2] / 2, ball[1].x, 1]
                ]).transpose()

                # Project points and estimate radius (projected size of ball radius)
                tProj, tRad, bProj, bRad = projectPoints(ballvec)

                # Draw
                cv2.circle(topCamImg[0], (int(tProj[0, 0]), int(tProj[1, 0])), tRad, 1, -1)
                cv2.circle(bottomCamImg[0], (int(bProj[0, 0]), int(bProj[1, 0])), bRad, 1, -1)

        if self.renderVar and self.agentVisID is not None and agent.id == self.agentVisID:

            # Visualization image size
            H = self.W // 2 - 50
            W = self.W // 2
            xOffs = 150
            img = np.zeros((H * 2, W * 2, 3)).astype('uint8')

            # Rotate FoV back for visualization
            vec1.rotate(-headAngle)
            vec2.rotate(-headAngle)

            # Draw
            cv2.line(img, (xOffs, H), (int(xOffs + vec1.x * 1000), int(H - vec1.y * 1000)), (255, 255, 0))
            cv2.line(img, (xOffs, H), (int(xOffs + vec2.x * 1000), int(H - vec2.y * 1000)), (255, 255, 0))

            # Draw all objects
            # Partially seen and distant objects are dim
            # Objects are drawn from the robot center
            for line in lineDets:
                color = (255, 255, 255) if line[0] == SightingType.Normal else (127, 127, 127)
                cv2.line(img, (int(xOffs + line[1].x), int(-line[1].y + H)),
                         (int(xOffs + line[2].x), int(-line[2].y + H)), color, self.lineWidth)

            if circleDets[0] != SightingType.NoSighting:
                color = (255, 0, 255) if circleDets[0] == SightingType.Normal else (127, 0, 127)
                cv2.circle(img, (int(xOffs + circleDets[1].x), int(-circleDets[1].y + H)), int(circleDets[2]), color,
                           self.lineWidth)

            for cross in crossDets:
                color = (255, 255, 255) if cross[0] == SightingType.Normal else (127, 127, 127)
                cv2.circle(img, (int(xOffs + cross[1].x), int(-cross[1].y + H)), int(cross[2]), color, -1)

            for cross in fieldCrossDets:
                color = (255, 255, 255) if cross[0] == SightingType.Normal else (127, 127, 127)
                cv2.circle(img, (int(xOffs + cross[1].x), int(-cross[1].y + H)), int(cross[2]), color, -1)

            for goal in goalDets:
                color = (255, 0, 0) if goal[0] == SightingType.Normal else (127, 0, 0)
                cv2.circle(img, (int(xOffs + goal[1].x), int(-goal[1].y + H)), int(goal[2]), color, -1)

            for i, rob in enumerate(robDets):
                color = (0, 255, 0) if rob[0] == SightingType.Normal else (0, 127, 0)
                cv2.circle(img, (int(xOffs + rob[1].x), int(-rob[1].y + H)), int(rob[2]), color, -1)

            for ball in ballDets:
                color = (0, 0, 255) if ball[0] == SightingType.Normal else (0, 0, 127)
                cv2.circle(img, (int(xOffs + ball[1].x), int(-ball[1].y + H)), int(ball[2]), color, -1)

            if self.renderMode == 'human':
                cv2.imshow(("Robot %d" % agent.id), img)
                c = cv2.waitKey(1)
                if c == 13:
                    cv2.imwrite("roboObs.png", img)
                    pygame.image.save(self.screen, "roboGame.png")
                if self.observationType == ObservationType.IMAGE:
                    cv2.imshow("Bottom", colorize(bottomCamImg))
                    cv2.imshow("Top", colorize(topCamImg))
            else:
                if self.observationType == ObservationType.IMAGE:
                    self.obsVis.append([topCamImg, bottomCamImg])
                else:
                    self.obsVis.append(img)

        if self.observationType == ObservationType.IMAGE:
            return np.concatenate((topCamImg, bottomCamImg))

        # Convert to numpy
        ballDets = np.array([[normalize(ball[1].x, self.standardNorm), normalize(ball[1].y, self.standardNorm),
                              normalizeAfterScale(ball[2], self.sizeNorm, self.ballRadius * 2), ball[3],
                              agent.id in self.closestID] for ball in ballDets]).astype('float32')
        robDets = np.array([[normalize(rob[1].x, self.standardNorm), normalize(rob[1].y, self.standardNorm),
                             normalizeAfterScale(rob[2], self.sizeNorm, Robot.totalRadius),
                             math.cos(rob[3]), math.sin(rob[3]), rob[4], rob[5]] for rob in robDets]).astype('float32')
        goalDets = np.array([convertToPolar(goal, self.standardNorm, self.goalPostRadius, self.sizeNorm, agent.team) for goal in
                             goalDets]).astype('float32')
        crossDets = np.array([convertToPolar(cross, self.standardNorm, self.penaltyRadius, self.sizeNorm, agent.team) for cross in
                              crossDets]).astype('float32')
        fieldCrossDets = np.array([convertToPolar(cross, self.standardNorm, self.penaltyRadius, self.sizeNorm, agent.team) + [math.cos(cross[5]), -math.sin(cross[5])] for cross in
                              fieldCrossDets]).astype('float32')
        selfDets = np.array([[normalizeAfterScale(agent.getPos()[0], self.standardNorm, self.meanX, agent.team),
                        normalizeAfterScale(agent.getPos()[1], self.standardNorm, self.meanY, agent.team),
                        math.cos(agent.getAngle(agent.team) + agent.headAngle),
                        math.sin(agent.getAngle(agent.team) + agent.headAngle),],]).astype('float32')
        lineDets = np.array([normalizeLine(line, self.standardNorm) for line in lineDets]).astype('float32')
        circleDets = np.array([[normalize(circleDets[1].x, self.standardNorm),
                                normalize(circleDets[1].y, self.standardNorm),
                                normalizeAfterScale(circleDets[2], self.sizeNorm * 0.1,
                                                    self.centerCircleRadius * 2)]]).astype('float32') \
            if circleDets[0] != SightingType.NoSighting else np.array([])


        return (ballDets, robDets), (goalDets, crossDets, fieldCrossDets, lineDets), (numLandMarks, robotsSeen, ballsSeen)

    def processSeens(self, observations):

        if self.observationType == ObservationType.FULL:
            return
        factor = 0.0025
        bfactor = 0.01

        for robID in range(self.nPlayers*2):
            lSeens = np.clip(np.array([float(obs[robID][2][0]) for obs in observations]).mean(), a_min=0.0, a_max=3.0)
            rSeens = np.clip(np.array([list(obs[robID][2][1]) for obs in observations]).sum(axis=0), a_min=0.0, a_max=2.0).sum()
            bSeens = np.clip(np.array([float(obs[robID][2][2]) for obs in observations]).sum(), a_min=0.0, a_max=3.0)
            if self.useObsRewards:
                self.obsRewards[robID] += (factor * (rSeens + lSeens) + bfactor * bSeens)