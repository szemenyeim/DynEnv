# coding=utf-8
#import time

import cv2
import pygame
import pymunk
import pymunk.pygame_util
from gym.spaces import Tuple, MultiDiscrete, Box, Dict, MultiBinary

from .Car import Car
from .Obstacle import Obstacle
from .Pedestrian import Pedestrian
from .Road import Road
from .cutils import *
from .environment_base import EnvironmentBase, RecoDescriptor, PredictionDescriptor, StateSpaceDescriptor


class DrivingEnvironment(EnvironmentBase):

    def __init__(self, nPlayers, render=False, observationType=ObservationType.PARTIAL, noiseType=NoiseType.REALISTIC,
                 noiseMagnitude=2, obs_space_cast=False, continuousActions=False):

        super().__init__(width=1700, height=1000, caption="Driving Environment", n_players=nPlayers,
                         max_players=10, n_time_steps=1, observation_type=observationType,
                         noise_type=noiseType, render=render, obs_space_cast=obs_space_cast,
                         noise_magnitude=noiseMagnitude, max_time=6000, step_iter_cnt=10)

        # Basic settings
        self.numTeams = 2
        self.nObjectType = 5
        self.continuousActions = continuousActions

        if self.observationType == ObservationType.IMAGE:
            print("Image type observation is not supported for this environment")
            exit(0)

        self._setup_normalization()
        self._setup_vision(0.4, 0.6)

        self.timeDiff = 10.0
        self.distThreshold = 100

        self._setup_observation_space()
        self._setup_action_space()
        self._setup_reconstruction_info()

        # Time rewards
        self.allFinished = False
        self.stepNum = self.maxTime / self.timeDiff

        # Episode rewards
        self._init_rewards()

        self._setup_scene()

        self._handle_collisions()

    def _setup_scene(self):
        self._create_roads()
        self._create_buildings()
        self._create_agents()
        self._create_pedestrians()
        self._create_obstacles()

    def _handle_collisions(self):
        # Ignore ped-ped and ped-obstacle collision
        self._add_collision_handler(CollisionType.Pedestrian, CollisionType.Pedestrian, self.ignore_collision)
        self._add_collision_handler(CollisionType.Pedestrian, CollisionType.Obstacle, self.ignore_collision)
        # Setup car-car collision
        self._add_collision_handler(CollisionType.Car, CollisionType.Car, self.carCrash)
        # Setup car-ped collision
        self._add_collision_handler(CollisionType.Car, CollisionType.Pedestrian, self.pedHit)
        # Setup car-obst collision
        self._add_collision_handler(CollisionType.Car, CollisionType.Obstacle, self.carHit)

    def _create_obstacles(self):
        self.obstacleNum = random.randint(10, 20)
        self.obstacles = self.createRandomObstacles()
        for obs in self.obstacles:
            self.space.add(obs.shape.body, obs.shape)

    def _create_pedestrians(self):
        self.pedestrianNum = random.randint(10, 20)
        self.pedestrians = self.createRandomPedestrians()
        for ped in self.pedestrians:
            self.space.add(ped.shape.body, ped.shape)

    def _create_agents(self):
        roadSel = np.random.randint(0, len(self.roads), self.nPlayers)
        endSel = np.random.randint(0, 2, self.nPlayers)
        goals = [self.roads[rd].points[end] for rd, end in zip(roadSel, endSel)]
        teams = np.random.randint(0, self.numTeams + 1, self.nPlayers)
        types = np.random.randint(0, len(Car.powers), self.nPlayers)  #
        spots = self.getUniqueSpots()
        self.agents = [Car(spot[0], spot[1], tp, team, goal) for spot, tp, team, goal in
                       zip(spots, types, teams, goals)]
        for car in self.agents:
            self.space.add(car.shape.body, car.shape)

    def _create_buildings(self):
        self.buildings = [
            Obstacle(pymunk.Vec2d(365.0, 200.0), 400, 225),
            Obstacle(pymunk.Vec2d(365.0, 800.0), 400, 225),
            Obstacle(pymunk.Vec2d(1385.0, 200.0), 400, 225),
            Obstacle(pymunk.Vec2d(1385.0, 800.0), 400, 225),
        ]
        for building in self.buildings:
            self.space.add(building.shape.body, building.shape)

    def _create_roads(self):
        self.roads = [
            Road(2, 35, [pymunk.Vec2d(875, 0), pymunk.Vec2d(875, 1000)]),
            Road(1, 35, [pymunk.Vec2d(0, 500), pymunk.Vec2d(1750, 500)]),
        ]
        self.laneNum = len(self.roads) + sum([r.nLanes * 2 for r in self.roads])

    def _init_rewards(self):
        self.episodeRewards = np.array([0.0, ] * self.nPlayers)
        self.episodePosRewards = np.array([0.0, ] * self.nPlayers)

    def get_full_obs(self):
        obs = [self.getFullState(car) for car in self.agents]
        obs = [([o[1], o[2], o[3]], [o[0], o[4]], (1, 1, 1)) for o in obs]
        return obs

    def get_agent_locs(self):
        return [self.getFullState(agent)[0][:, [0, 1, 2, 3]] for agent in self.agents]

    def _setup_reconstruction_info(self):

        # components
        pos_xy = Box(-self.mean * 2, +self.mean * 2, shape=(2,))
        orientation = Box(-1.0, +1.0, shape=(2,))
        size = Box(-10, 10, shape=(2,))
        confidence = MultiBinary(1)

        # Self
        self_state = StateSpaceDescriptor(1, Dict({"position": pos_xy,
                                                   "orientation": orientation,
                                                   "size": size,
                                                   "confidence": confidence,
                                                   }))
        self_pred = PredictionDescriptor(numContinuous=4, contIdx=[2, 3, 4, 5])

        # Car
        car_state = StateSpaceDescriptor(4, Dict({"position": pos_xy,
                                                  "orientation": orientation,
                                                  "size": size,
                                                  "confidence": confidence,
                                                  }))

        car_pred = PredictionDescriptor(numContinuous=4, contIdx=[2, 3, 4, 5])

        # Obstacle
        obstacle_state = StateSpaceDescriptor(4, Dict({"position": pos_xy,
                                                       "size": size,
                                                       "confidence": confidence,
                                                       }))
        obs_pred = PredictionDescriptor(numContinuous=2, contIdx=[2, 3])

        # Pedestrian
        ped_state = StateSpaceDescriptor(6, Dict({"position": pos_xy,
                                                  "confidence": confidence, }))
        ped_pred = PredictionDescriptor(numContinuous=0)

        self.recoDescriptor = RecoDescriptor(featureGridSize=(10, 17),
                                             fullStateSpace=[self_state, car_state, obstacle_state, ped_state],
                                             targetDefs=[self_pred, car_pred, obs_pred, ped_pred])

    def _setup_action_space(self):
        if self.continuousActions:
            self.action_space = Tuple((Box(-3, +3, shape=(2,)),))
        else:
            self.action_space = Tuple((MultiDiscrete([3, 3]),))

    def _create_observation_space(self):
        # components
        pos_xy = Box(-self.mean * 2, +self.mean * 2, shape=(2,))
        width_height = Box(-10, 10, shape=(2,))
        orientation = Box(-1, 1, shape=(2,))
        type = Box(-1, 1, shape=(1,))

        # spaces for all cases
        self_space = Dict({
            "position": pos_xy,
            "orientation": orientation,
            "width_height": width_height,
            "goal_position": pos_xy,
            "finished": MultiBinary(1)
        })
        car_space = Dict({
            "position": pos_xy,
            "orientation": orientation,
            "width_height": width_height,
            "finished": MultiBinary(1)
        })
        obstacle_space = Dict({
            "position": pos_xy,
            "orientation": orientation,
            "width_height": width_height,
        })
        pedestrian_space = Dict({
            "position": pos_xy,
        })

        if self.observationType == ObservationType.FULL:
            lane_space = Dict({
                "points": Box(-self.mean * 2, self.mean * 2, shape=(4,)),
                "type": type
            })
            obstacle_space = Dict({
                "position": pos_xy,
                "width_height": width_height,
            })
        else:
            lane_space = Dict({
                "signed_distance": Box(-self.mean * 2, self.mean * 2, shape=(1,)),
                "orientation": orientation,
                "type": type
            })

        self.observation_space = Tuple([
            Tuple([
                car_space,
                obstacle_space,
                pedestrian_space,
                ]),
            Tuple([
                self_space,
                lane_space
            ])
        ])

    def _setup_normalization(self):
        self.mean = 5.0 if ObservationType.PARTIAL else 1.0
        self.normX = self.mean * 2 / self.W
        self.normY = self.mean * 2 / self.H
        self.normW = 1.0 / 7.5
        self.normH = 1.0 / 15
        self.standardNormX = 0.5 / (self.W + 100)
        self.standardNormY = 0.5 / (self.H + 100)
        self.standardNormW = 1.0 / 15
        self.standardNormH = 1.0 / 25

    def get_class_specific_args(self):
        return [self.continuousActions]

    def step(self, actions):
        #t1 = time.clock()

        # Setup reward and state variables
        self.teamReward = 0.0
        self.carRewards = np.array([0.0, ] * self.nPlayers)
        self.carPosRewards = np.array([0.0, ] * self.nPlayers)
        finished = False
        observations = []

        # Run simulation for 100 ms (time for every action)
        for i in range(self.stepIterCnt):

            # Sanity check
            if actions.shape != (len(self.agents), 2):
                raise Exception("Error: There must be 2 actions for every car")

            # Car loop
            for action, car in zip(actions, self.agents):
                # Apply action as first step
                if i % 10 == 0:
                    self.processAction(action, car)

                # Update cars
                self.tick(car)

            # Update pedestrians
            [self.move(ped) for ped in self.pedestrians]

            # Run simulation
            self.space.step(1 / self.timeStep)

            self.elapsed += 1
            allFinished = all([car.finished and not car.crashed for car in self.agents])
            if not self.allFinished:
                if allFinished:
                    self.allFinished = True
                    self.teamReward += (self.maxTime - self.elapsed) / 100
                elif self.elapsed >= self.maxTime:
                    self.teamReward -= 0

            # Get observations every 100 ms
            if i % 10 == 9:
                if self.observationType == ObservationType.FULL:
                    observations.append(self.get_full_obs())
                else:
                    observations.append([self.getAgentVision(car) for car in self.agents])

            if self.renderVar:
                self._render_internal()

        # Reward finishing
        self.carRewards += self.teamReward
        self.episodeRewards += self.carRewards

        self.carPosRewards += max(0.0, self.teamReward)
        self.episodePosRewards += self.carPosRewards

        info = {'Full State': self.getFullState()}
        info['Recon States'] = [self.getFullState(car) for car in self.agents]

        # Episode finishing
        if self.elapsed >= self.maxTime:
            finished = True
            info['episode_r'] = self.episodeRewards
            info['episode_p_r'] = self.episodePosRewards
            info['episode_o_r'] = [0,]*self.nPlayers
            info['episode_g'] = [sum([car.finished and not car.crashed for car in self.agents]),
                                 sum([car.crashed for car in self.agents])]
            # print(self.episodeRewards)

        #t2 = time.clock()
        # print((t2 - t1) * 1000)

        return observations, self.carRewards, finished, info

    def drawStaticObjects(self):

        # Gray screen
        self.screen.fill((50, 50, 50))

        # Draw roads
        for road in self.roads:

            # Walkway edges
            for line in road.Walkways:
                pygame.draw.line(self.screen, (255, 255, 0), line[0], line[1], 1)

            # Draw lanes
            for i, line in enumerate(road.Lanes):

                # Default params
                color = (255, 255, 255)
                thickness = 1

                # Edges with red
                if i == 0 or i == road.nLanes * 2:
                    color = (255, 0, 0)
                # Middle double thick
                elif i == road.nLanes:
                    thickness = 2

                # Draw
                pygame.draw.line(self.screen, color, line[0], line[1], thickness)

        # draw everything else
        self.space.debug_draw(self.draw_options)

    # Process actions
    def processAction(self, action, car):

        # Get actions
        if not self.continuousActions:
            acc = action[0] - 1
            steer = (action[1] - 1) * 2

        # Sanity checks
        if np.abs(acc) > 3:
            raise Exception("Error: Acceleration must be between +/-3")
        if np.abs(steer) > 3:
            raise Exception("Error: Steering must be between +/-3")

        # Apply actions to car
        car.accelerate(acc.item(), self.continuousActions)
        if steer != 0:
            car.turn(steer.item())

    # Update car status
    def tick(self, car):

        # Get params
        car.position = LanePosition.OffRoad
        index = self.agents.index(car)

        # Get car position relative to roads
        for road in self.roads:
            rPos = road.isPointOnRoad(car.getPos(), car.getAngle())
            car.position = min(car.position, rPos)

        # Reward for getting closer
        diff = (car.prevPos - car.goal).length - (car.getPos() - car.goal).length
        if not car.finished:
            self.carRewards[index] += diff / 50
            self.carPosRewards[index] += max(0.0, diff / 50)

        # Update previous position
        car.prevPos = car.getPos()

        # crash for leaving road
        if car.position >= LanePosition.OverRoad:
            if not car.finished:
                # If reached goal finish and add reward
                if car.position == LanePosition.OverRoad and (car.getPos() - car.goal).length < self.distThreshold:
                    car.position = LanePosition.AtGoal
                    car.finished = True
                    self.carRewards[index] += (self.maxTime - self.elapsed) / 100
                    self.carPosRewards[index] += (self.maxTime - self.elapsed) / 100
                    car.shape.body.velocity_func = friction_car_crashed
                else:
                    car.crash()
                    self.carRewards[index] -= car.shape.body.velocity.length / 5
        # Add small punichment for being in opposite lane
        elif car.position == LanePosition.InOpposingLane:
            if not car.finished:
                self.carRewards[index] -= car.shape.body.velocity.length / 10000

        # Stop car outside the field
        if car.prevPos.x >= self.W + 50:
            car.shape.body.velocity = Vec2d(0, 0)
            car.shape.body.position.x = self.W + 49
        if car.prevPos.x <= -50:
            car.shape.body.velocity = Vec2d(0, 0)
            car.shape.body.position.x = - 49
        if car.prevPos.y >= self.H + 50:
            car.shape.body.velocity = Vec2d(0, 0)
            car.shape.body.position.y = self.H + 49
        if car.prevPos.y <= -50:
            car.shape.body.velocity = Vec2d(0, 0)
            car.shape.body.position.y = - 49

    # Update function for pedestrians
    def move(self, pedestrian):

        # Dead pedestrians don't move
        if not pedestrian.dead:

            # Get pedestrian status
            isOffRoad = self.isOffRoad(pedestrian.getPos())
            isOut = self.isOut(pedestrian.getPos())

            # If there is still time left for the current movement
            if pedestrian.moving > 0:

                # Decrease timer
                pedestrian.moving = max(0, pedestrian.moving - self.timeDiff)

                # If the pedestrian is crossing the road
                if pedestrian.crossing:

                    # If the pedestrian finished crossing
                    if not pedestrian.beginCrossing and isOffRoad:
                        # Reset everything and stop
                        pedestrian.moving = 0
                        pedestrian.crossing = False
                        pedestrian.shape.body.velocity = pymunk.Vec2d(0, 0)

                    # If the pedestrian just started crossing and got on the road
                    elif pedestrian.beginCrossing and not isOffRoad:
                        # Set begin to false
                        pedestrian.beginCrossing = False
                # If pedestrian walked out, stop and get new direction in next tick
                if isOut:
                    pedestrian.moving = 0
                    pedestrian.shape.body.velocity = pymunk.Vec2d(0, 0)
            # If movement expired
            else:
                # If the pedestrian is not crossing, get new movement
                if not pedestrian.crossing:

                    # Get random time and speed
                    pedestrian.moving = random.randint(5000, 30000)
                    speed = random.randint(-2, 2)
                    dir = pedestrian.direction

                    # If pedestrian is on the road
                    if not isOffRoad:
                        # Setup crossing params
                        pedestrian.crossing = True
                        pedestrian.beginCrossing = False

                        # Don't stop on the road, duh!
                        if speed == 0:
                            speed = 2
                    # If pedestrian is out, choose direction towards the middle
                    elif isOut:
                        dir = -pedestrian.direction if self.isOut(
                            pedestrian.getPos() + pedestrian.direction) else pedestrian.direction
                    # Otherwise cross the road with 5% chance
                    elif random.random() < 0.05:

                        # Setup normal crossing
                        pedestrian.crossing = True
                        pedestrian.beginCrossing = True

                        # Get direction
                        dir = pedestrian.normal if pedestrian.side else -pedestrian.normal

                        # Change side
                        pedestrian.side = 0 if pedestrian.side else 1

                        # Set non-zero speed
                        speed = random.randint(1, 2)

                    # Set new speed
                    pedestrian.shape.body.velocity = pedestrian.speed * dir * speed
                # If pedestrian was crossing but now is off the road, reset crossing vars
                elif isOffRoad:
                    pedestrian.crossing = False
                    pedestrian.beginCrossing = False

    # Is point off the road
    def isOffRoad(self, point):

        # Default
        position = LanePosition.OffRoad

        # Get position relative to the roads
        for road in self.roads:
            rPos = road.isPointOnRoad(point, 0)
            position = min(position, rPos)

        # Return
        return position >= LanePosition.OverRoad

    # Is point out of the field
    def isOut(self, pos):
        return pos.x <= 0 or pos.y <= 0 or pos.x >= self.W or pos.y >= self.H

    # Get unique sports on the road
    def getUniqueSpots(self):

        # 5 spots for every lane (road.nLanes is the lanes in a single direction)
        roadSpots = np.array([10 * road.nLanes for road in self.roads])

        # IDs and offsets for indexing
        roadSpotsIDs = np.array([sum(roadSpots[0:i + 1]) for i in range(len(self.roads))])
        roadSpotsOffs = np.array([sum(roadSpots[0:i]) for i in range(len(self.roads))])

        # Get random spots
        nSpots = sum(roadSpots)
        spotIDs = np.random.permutation(nSpots)[:self.nPlayers]

        # Get the ID of the road for each spot
        roadIDs = np.array([np.where(spotID < roadSpotsIDs)[0][0] for spotID in spotIDs])

        # Subtract road offset
        spotIDs -= roadSpotsOffs[roadIDs]

        # Get index of the lane and the spot
        laneIDs = spotIDs // 5
        spotIDs = spotIDs % 5

        # Get the actual spots
        return [self.roads[roadID].getSpot(laneID, spotID) for roadID, laneID, spotID in zip(roadIDs, laneIDs, spotIDs)]

    # Create random pedestrians
    def createRandomPedestrians(self):

        # Select roads, sides
        roadIds = np.random.randint(0, len(self.roads), self.pedestrianNum)
        sideIds = np.random.randint(0, 2, self.pedestrianNum)

        # Create random placements along the length and width of the walkways
        lenOffs = np.random.rand(self.pedestrianNum)
        widthOffs = np.random.rand(self.pedestrianNum) / 2 + 0.25

        # Get actual pedestrians
        return [Pedestrian(self.roads[road].getWalkSpot(side, length, width), self.roads[road], side) for
                road, side, length, width in zip(roadIds, sideIds, lenOffs, widthOffs)]

    # Create random obstacles
    def createRandomObstacles(self):

        # Select roads, sides
        roadIds = np.random.randint(0, len(self.roads), self.obstacleNum)
        sideIds = np.random.randint(0, 2, self.obstacleNum)

        # Create random placements along the length and width of the walkways
        lenOffs = np.random.rand(self.obstacleNum)
        widthOffs = np.random.rand(self.obstacleNum) / 2 + 0.25

        # Get actual obstacles
        obs = [Obstacle(self.roads[road].getWalkSpot(side, length, width), 10, 10) for road, side, length, width in
               zip(roadIds, sideIds, lenOffs, widthOffs)]

        # Remove the ones on the roads
        return [ob for ob in obs if self.isOffRoad(ob.getPos())]

    # Don't handle pedestrian obstacle collision
    def ignore_collision(self, arbiter, space, data):
        return False

    # Car collision
    def carCrash(self, arbiter, space, data):

        # Get cars
        car1 = next(car for car in self.agents if (car.shape == arbiter.shapes[0]))
        car2 = next(car for car in self.agents if (car.shape == arbiter.shapes[1]))

        # Punish
        index1 = self.agents.index(car1)
        index2 = self.agents.index(car2)

        v1l = car1.shape.body.velocity.length / 5
        v2l = car2.shape.body.velocity.length / 5

        if not car1.crashed:
            self.carRewards[index1] -= v1l
        if not car2.crashed:
            self.carRewards[index2] -= v2l

        # Punish car in the wrong lane extra
        if car1.position != LanePosition.InRightLane and not car1.crashed:
            self.carRewards[index1] -= v1l

        if car2.position != LanePosition.InRightLane and not car2.crashed:
            self.carRewards[index2] -= v2l

        # If both in the rights lane
        if car1.position == LanePosition.InRightLane and car2.position == LanePosition.InRightLane:

            # Get velocities and relative position
            v1 = arbiter.shapes[0].body.velocity
            v2 = arbiter.shapes[1].body.velocity
            p1 = car1.getPos()
            p2 = car2.getPos()
            dp = p1 - p2

            # Car is responsible if moving towards the other one
            if v1.length > 1 and math.cos(angle(dp) - angle(v1)) < -0.4 and not car1.crashed:
                self.carRewards[index1] -= v1l

            if v2.length > 1 and math.cos(angle(dp) - angle(v2)) > 0.4 and not car2.crashed:
                self.carRewards[index2] -= v2l

        # Crash them
        car1.crash()
        car2.crash()

        return True

    # Handle car hitting pedestrian
    def pedHit(self, arbiter, space, data):

        # Get objects
        car = next(car for car in self.agents if (car.shape == arbiter.shapes[0]))
        ped = next(ped for ped in self.pedestrians if (ped.shape == arbiter.shapes[1]))

        # Get velocity
        v1 = arbiter.shapes[0].body.velocity
        v1l = v1.length
        if v1l > 1:

            # Blergh
            ped.die()

            # Get relative positions
            p1 = car.getPos()
            p2 = ped.getPos()
            dp = p1 - p2

            # Crash car if it actually hit pedestrian
            if math.cos(angle(dp) - angle(v1)) < -0.4 and not car.finished:
                car.crash()
                index = self.agents.index(car)
                self.carRewards[index] -= v1l / 5
        else:
            return False

        return True

    # Car-obstacle crash
    def carHit(self, arbiter, space, data):

        # Get car
        car = next(car for car in self.agents if (car.shape == arbiter.shapes[0]))

        # Punish
        index = self.agents.index(car)
        if not car.finished:
            self.carRewards[index] -= car.shape.body.velocity.length / 5

        # crash car
        car.crash()

        return True

    # Gett correct state
    def getFullState(self, agent=None):

        # Get lanes
        lanes = []
        for l in self.roads:
            lanes += [[normalize(l.Lanes[i - l.nLanes][0].x, self.standardNormX),
                       normalize(l.Lanes[i - l.nLanes][0].y, self.standardNormY),
                       normalize(l.Lanes[i - l.nLanes][1].x, self.standardNormX),
                       normalize(l.Lanes[i - l.nLanes][1].y, self.standardNormY),
                       (1 if abs(i) == l.nLanes else (-1 if i == 0 else 0))] for i in range(-l.nLanes, l.nLanes + 1)]

        # If complete state
        if agent is None:

            # Just add cars
            state = [
                np.array([[normalize(c.getPos().x, self.standardNormX),
                           normalize(c.getPos().y, self.standardNormY),
                           math.cos(c.getAngle()),
                           math.sin(c.getAngle()),
                           normalize(c.width, self.standardNormW, mean=0.5),
                           normalize(c.height, self.standardNormH, mean=0.5),
                           c.finished] for c in self.agents]).astype('float32'),
            ]
        # Otherwise add self observation separately
        else:
            state = [
                np.array([[normalize(agent.getPos().x, self.standardNormX),
                          normalize(agent.getPos().y, self.standardNormY),
                          math.cos(agent.getAngle()),
                          math.sin(agent.getAngle()),
                          normalize(agent.width, self.standardNormW, mean=0.5),
                          normalize(agent.height, self.standardNormH, mean=0.5),
                          normalize(agent.goal.x, self.standardNormX),
                          normalize(agent.goal.y, self.standardNormY),
                          agent.finished],]).astype('float32'),

                np.array([[normalize(c.getPos().x, self.standardNormX),
                           normalize(c.getPos().y, self.standardNormY),
                           math.cos(c.getAngle()),
                           math.sin(c.getAngle()),
                           normalize(c.width, self.standardNormW, mean=0.5),
                           normalize(c.height, self.standardNormH, mean=0.5),
                           c.finished] for c in self.agents if c != agent]).astype('float32'),
            ]

        # Add obstacles, pedestrians and lanes
        state += [
            np.array([[normalize(o.getPos().x, self.standardNormX),
                       normalize(o.getPos().y, self.standardNormY),
                       normalize(o.width, self.standardNormW, mean=0.5),
                       normalize(o.height, self.standardNormH, mean=0.5)]
                      for o in self.obstacles]).astype('float32'),

            np.array([[normalize(p.getPos().x, self.standardNormX),
                       normalize(p.getPos().y, self.standardNormY)]
                      for p in self.pedestrians]).astype('float32'),

            np.array(lanes).astype('float32')
        ]

        return state

    # Get car observation
    def getAgentVision(self, agent):

        ang = agent.getAngle()

        # Get detections within radius
        selfDet = [SightingType.Normal, agent.getPos(), math.cos(ang), math.sin(ang), agent.getPoints(), agent.width,
                   agent.height, agent.goal, agent.finished]
        carDets = [isSeenInRadius(c.getPos(), c.getPoints(), c.getAngle(), selfDet[1], ang, self.maxVisDist[0],
                                  self.maxVisDist[1])
                   + [c.width, c.height] + [c.finished, ] for c in self.agents if c != agent]
        obsDets = [
            isSeenInRadius(o.getPos(), o.points, 0, selfDet[1], ang, self.maxVisDist[0], self.maxVisDist[1]) + [o.width,
                                                                                                                o.height]
            for o in self.obstacles]
        buildDets = [isSeenInRadius(b.getPos(), b.points, 0, selfDet[1], ang, 20000000, 20000000) for b in
                     self.buildings]  # Buildings are always seen
        pedDets = [isSeenInRadius(p.getPos(), None, 0, selfDet[1], ang, self.maxVisDist[0], self.maxVisDist[1]) for p in
                   self.pedestrians]
        laneDets = np.concatenate([r.getCarLaneDistances(selfDet[1], ang) for r in self.roads])
        '''laneDets = []
        for l in self.roads:
            laneDets += [getLineInRadius(l.Lanes[i + l.nLanes], selfDet[1], ang, self.maxVisDist[0])
                         + [(1 if abs(i) == l.nLanes else (-1 if i == 0 else 0)), ]
                         for i in range(-l.nLanes, l.nLanes + 1)]'''

        # Remove objects not seen (this is to reduce computation down the road)
        carDets = [c for i, c in enumerate(carDets) if c[0] != SightingType.NoSighting]
        pedDets = [ped for i, ped in enumerate(pedDets) if ped[0] != SightingType.NoSighting]
        obsDets = [obs for i, obs in enumerate(obsDets) if obs[0] != SightingType.NoSighting]
        laneDets = [lane for i, lane in enumerate(laneDets) if lane[0] != SightingType.NoSighting]

        # Get objects occluded by buildings
        buildCarInter = [max([doesInteractPoly(c, b, 0) for b in buildDets]) for c in carDets]
        buildPedInter = [max([doesInteractPoly(p, b, 0) for b in buildDets]) for p in pedDets]
        buildObsInter = [max([doesInteractPoly(o, b, 0) for b in buildDets]) for o in obsDets]

        # Remove occluded objects (this is to reduce computation down the road)
        carDets = [c for i, c in enumerate(carDets) if buildCarInter[i] != InteractionType.Occlude]
        pedDets = [ped for i, ped in enumerate(pedDets) if buildPedInter[i] != InteractionType.Occlude]
        obsDets = [obs for i, obs in enumerate(obsDets) if buildObsInter[i] != InteractionType.Occlude]

        # Get pedestrian-car and pedestrian-obstacle interactions
        carPedInter = [max([doesInteractPoly(p, c, 400) for c in carDets]) for p in pedDets] if carDets else [
                                                                                                                 InteractionType.NoInter, ] * len(
            pedDets)
        obsPedInter = [max([doesInteractPoly(p, o, 400) for o in obsDets]) for p in pedDets] if obsDets else [
                                                                                                                 InteractionType.NoInter, ] * len(
            pedDets)
        pedInter = max(carPedInter, obsPedInter)

        # set occluded pedestrians to NoSighting
        [filterOcclude(p, pedInter[i]) for i, p in enumerate(pedDets)]

        # Add noise: Self, Car, Obs, Ped and Lane
        addNoiseRect(selfDet, self.noiseType, InteractionType.NoInter, self.noiseMagnitude, self.randBase,
                     self.maxVisDist[1])
        [addNoiseRect(c, self.noiseType, InteractionType.NoInter, self.noiseMagnitude, self.randBase,
                      self.maxVisDist[1], True) for i, c in enumerate(carDets)]
        [addNoiseRect(ped, self.noiseType, pedInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[0]) for
         i, ped in enumerate(pedDets)]
        [addNoiseRect(obs, self.noiseType, InteractionType.NoInter, self.noiseMagnitude, self.randBase,
                      self.maxVisDist[1], True) for i, obs in enumerate(obsDets)]
        [addNoiseLane(lane, self.noiseType, self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i, lane in
         enumerate(laneDets)]

        # Cars and obstacles might by misclassified - move them in the other list
        for c in carDets:
            if c[0] == SightingType.Misclassified:
                obsDets.append((SightingType.Normal, c[1], c[2], c[3], c[4], c[5], c[6]))
        for obs in obsDets:
            if obs[0] == SightingType.Misclassified:
                carDets.append((SightingType.Normal, obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], False))

        # Random false positives
        for i in range(10):
            if random.random() < self.randBase:

                # Class
                c = random.randint(0, 5)

                # Distance and angle
                d = random.random() * self.maxVisDist[1]
                a1 = random.random() * 2 * math.pi
                pos = pymunk.Vec2d(d, 0)
                pos.rotate(a1)

                # Object angle
                angle = random.random() * 2 * math.pi
                co = math.cos(angle)
                si = math.sin(angle)

                # car or obstacle
                if c <= 1:

                    # Width and height
                    w = random.random() * 5 + 5
                    h = random.random() * 10 + 5

                    # create corners
                    obs = [Vec2d(h, w), Vec2d(-h, w), -Vec2d(h, w), Vec2d(h, -w)]
                    [ob.rotate(angle) for ob in obs]
                    obs = [ob + pos for ob in obs]

                    # Add objects
                    if c == 0:
                        carDets.insert(len(carDets),
                                       [SightingType.Normal, pos, co, si, obs, w, h, False])
                    else:
                        obsDets.insert(len(obsDets),
                                       [SightingType.Normal, pos, co, si, obs, w, h])
                # Add pedestrian
                elif c == 2:
                    pedDets.insert(len(pedDets),
                                   [SightingType.Normal, pos])
                # Add lane
                elif c == 3:
                    # Get second point
                    a = (random.random() - 0.5) * math.pi * 2
                    c = math.cos(a)
                    s = math.sin(a)
                    dist = random.random() * self.W // 2

                    # Add lane
                    laneDets.insert(len(laneDets),
                                    [SightingType.Normal, dist, c, s, random.randint(-1, 1)])

        # FP Pedestrians near cars and obstacles
        if self.noiseType == NoiseType.REALISTIC:
            for c in carDets:
                if c[0] == SightingType.Normal and random.random() < self.randBase * 10 and c[1].length < 250:
                    offset = pymunk.Vec2d(2 * random.random() - 1.0, 2 * random.random() - 1.0) * 10
                    pedDets.insert(len(pedDets),
                                   [SightingType.Normal, c[1] + offset, [], 0])

        # Remove occlusion and misclassified originals
        carDets = [c for i, c in enumerate(carDets) if
                   c[0] != SightingType.NoSighting and c[0] != SightingType.Misclassified]
        pedDets = [ped for i, ped in enumerate(pedDets) if ped[0] != SightingType.NoSighting]
        obsDets = [obs for i, obs in enumerate(obsDets) if
                   obs[0] != SightingType.NoSighting and obs[0] != SightingType.Misclassified]
        laneDets = [lane for i, lane in enumerate(laneDets) if lane[0] != SightingType.NoSighting]

        if self.renderVar and self.agentVisID is not None and self.agents.index(agent) == self.agentVisID:

            # Visualization image size
            H = self.H // 2
            W = self.W // 2
            img = np.zeros((H * 2, W * 2, 3)).astype('uint8')

            # Draw all objects
            # Partially seen and distant objects are dim
            # Objects are drawn from the robot center

            # Draw buildings first
            for build in buildDets:
                color = (200, 200, 200)
                points = build[4]
                cv2.fillConvexPoly(img, np.array([(int(p.x + W), int(-p.y + H)) for p in points]), color)

            # draw lane (color based on type)
            for lane in laneDets:
                color = (127, 127, 255) if lane[4] == -1 else ((127, 255, 127) if lane[4] == 1 else (255, 255, 255))
                if lane[0] != SightingType.Normal:
                    color = (color[0] // 2, color[1] // 2, color[2] // 2)

                # Get line points from params
                a = lane[2]
                b = -lane[3]
                rho = -lane[1] / Road.laneScaleFactor
                #print(lane[1], lane[4])
                x0 = b * rho
                y0 = a * rho
                pt1 = (int(np.round(x0 - 5000 * a) + W), int(H - np.round(y0 + 5000 * b)))
                pt2 = (int(np.round(x0 + 5000 * a) + W), int(H - np.round(y0 - 5000 * b)))

                cv2.line(img, pt1, pt2, color, 30)

            # draw self
            color = (0, 255, 255)
            points = [p - agent.getPos() for p in selfDet[4]]
            cv2.fillConvexPoly(img, np.array([(int(p.x + W), int(-p.y + H)) for p in points]), color)

            # draw cars
            for c in carDets:
                color = (0, 255, 0) if c[0] == SightingType.Normal else (0, 127, 0)
                points = c[4]
                cv2.fillConvexPoly(img, np.array([(int(p.x + W), int(-p.y + H)) for p in points]), color)

            # draw obstacles
            for obs in obsDets:
                color = (200, 200, 200) if obs[0] == SightingType.Normal else (100, 100, 100)
                points = obs[4]
                cv2.fillConvexPoly(img, np.array([(int(p.x + W), int(-p.y + H)) for p in points]), color)

            # draw pedestrians
            for ped in pedDets:
                color = (255, 255, 0) if ped[0] == SightingType.Normal else (127, 127, 0)
                point = ped[1]
                cv2.circle(img, (int(point.x + W), int(-point.y + H)), 5, color, -1)

            if self.renderMode == 'human':
                cv2.imshow(("Car %d" % self.agents.index(agent)), img)
                c = cv2.waitKey(1)
                if c == 13:
                    cv2.imwrite("drivingObs.png", img)
                    pygame.image.save(self.screen, "drivingGame.png")
            else:
                self.obsVis.append(img)

        # Convert to numpy
        selfDet = np.array([[normalize(selfDet[1].x, self.normX, self.mean),
                             normalize(selfDet[1].y, self.normY, self.mean), selfDet[2], selfDet[3],
                             normalize(selfDet[5], self.normW, 0.5), normalize(selfDet[6], self.normH, 0.5),
                             normalize(selfDet[7].x, self.normX, self.mean),
                             normalize(selfDet[7].y, self.normY, self.mean), selfDet[8]], ]).astype('float32')
        carDets = np.array([[normalize(c[1].x, self.normX), normalize(c[1].y, self.normY), c[2], c[3],
                             normalize(c[5], self.normW, 0.5), normalize(c[6], self.normH, 0.5), c[7]] for c in
                            carDets]).astype('float32')
        obsDets = np.array([[normalize(obs[1].x, self.normX), normalize(obs[1].y, self.normY), obs[2], obs[3],
                             normalize(obs[5], self.normW, 0.5), normalize(obs[6], self.normH, 0.5)] for obs in
                            obsDets]).astype('float32')
        pedDets = np.array(
            [[normalize(ped[1].x, self.normX), normalize(ped[1].y, self.normY)] for ped in pedDets]).astype('float32')
        laneDets = np.array([[lane[1], lane[2], lane[3], lane[4]] for lane in laneDets]).astype(
            'float32')

        # return
        return (carDets, obsDets, pedDets), (selfDet, laneDets), (1,1,1)

    # Print env params
    def __str__(self):
        return "Driving Simulation Environment\n" \
               "Created by Márton Szemenyei\n\n" \
               "Parameters:\n" \
               "    nPlayers: Number of cars\n" \
               "    render: Wether to render the environment using pyGame\n" \
               "    observationType: Choose between full state and partial observation types (2D images are not supported)\n" \
               "    noiseType: Choose between random and realistic noise\n" \
               "    noiseMagnitude: Set the amount of the noise between 0-5\n" \
               "Actions:\n" \
               "    Gas/Break: 1,0,-1\n" \
               "    Turn: 2,1,0,-1,-2\n" \
               "Return values:\n" \
               "    Full state: Contains the correct car info for all\n" \
               "        Cars [position, corner points, angle]\n" \
               "        Obstacles [position, corners]\n" \
               "        Pedestrians [position]\n" \
               "        Lanes [point1, point2, type]\n" \
               "    Observations: Contains car observations (in the same order as the cars are in the full state):\n" \
               "        Self detection: [sightingType, position, corners, angle, goal]\n" \
               "        Car detections: [sightingType, position, corners, angle]\n" \
               "        Obstacle detections: [sightingType, position, corners]\n" \
               "        Pedestrian detections: [sightingType, position]\n" \
               "        Lane detections: [sightingType, point1, point2, type]\n" \
               "    Team rewards: rewards for each team\n" \
               "    Car rewards: rewards for each car (in the same order as in state)\n" \
               "    Finished: Game over flag"
