from .Car import Car
from .Pedestrian import Pedestrian
from .Obstacle import Obstacle
from .Road import Road
from .utils import ObservationType, NoiseType, CollisionType
from .cutils import LanePosition
import cv2
import math
import time
import pymunk
import pymunk.pygame_util
import pygame
import numpy as np
import random

class DrivingEnvironment(object):

    def __init__(self,nPlayers,render=False,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic, noiseMagnitude = 2):

        # Basic settings
        self.observationType = observationType
        self.noiseType = noiseType
        self.maxPlayers = 10
        self.nPlayers = min(nPlayers,self.maxPlayers)
        self.render = render
        self.numTeams = 2

        # Noise
        self.noiseMagnitude = noiseMagnitude

        # Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.timeStep = 100.0
        self.timeDiff = 10.0
        self.distThreshold = 10

        # Setup scene
        self.W = 1750
        self.H = 1000

        # Time rewards
        self.maxTime = 2000
        self.elapsed = 0
        self.allFinished = False

        # Setup roads
        self.roads = [
            Road(2,35,[pymunk.Vec2d(875,0),pymunk.Vec2d(875,1000)]),
            Road(1,35,[pymunk.Vec2d(0,500),pymunk.Vec2d(1750,500)]),
        ]

        self.obstacles = [
            Obstacle(pymunk.Vec2d(365,200),400,225),
            Obstacle(pymunk.Vec2d(365,800),400,225),
            Obstacle(pymunk.Vec2d(1385,200),400,225),
            Obstacle(pymunk.Vec2d(1385,800),400,225),
        ]

        # Add cars
        roadSel = np.random.randint(0,len(self.roads), self.nPlayers)
        endSel = np.random.randint(0,2,self.nPlayers)
        goals = [self.roads[rd].points[end] for rd,end in zip(roadSel,endSel)]
        teams = np.random.randint(0,self.numTeams+1,self.nPlayers)
        types = np.random.randint(0,len(Car.powers),self.nPlayers)
        spots = self.getUniqueSpots()
        self.cars = [Car(spot[0],spot[1],tp,team,goal) for spot,tp,team,goal in zip(spots,types,teams,goals)]
        for car in self.cars:
            self.space.add(car.shape.body,car.shape)

        # Add random pedestrians
        self.pedestrianNum = 40
        self.pedestrians = self.createRandomPedestrians()
        for ped in self.pedestrians:
            self.space.add(ped.shape.body,ped.shape)

        # Add random obstacles
        self.obstacleNum = 20
        self.obstacles += self.createRandomObstacles()
        for obs in self.obstacles:
            self.space.add(obs.shape.body,obs.shape)

        # Setup ped-ped and ped-obstacle collision (ignore)
        h = self.space.add_collision_handler(
            CollisionType.Pedestrian,
            CollisionType.Pedestrian)
        h.begin = self.pedCollision
        h = self.space.add_collision_handler(
            CollisionType.Pedestrian,
            CollisionType.Obstacle)
        h.begin = self.pedCollision

        # Setup car-car collision
        h = self.space.add_collision_handler(
            CollisionType.Car,
            CollisionType.Car)
        h.begin = self.carCrash

        # Setup car-ped collision
        h = self.space.add_collision_handler(
            CollisionType.Car,
            CollisionType.Pedestrian)
        h.begin = self.pedHit

        # Setup car-obst collision
        h = self.space.add_collision_handler(
            CollisionType.Car,
            CollisionType.Obstacle)
        h.begin = self.carHit

        # Render options
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Driving Environment")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def step(self,actions):
        t1 = time.clock()

        # Setup reward and state variables
        self.teamReward = 0
        self.carRewards = [0,] * self.nPlayers
        observations = []
        finished = False

        # Run simulation for 500 ms (time for every action)
        for i in range(50):

            # Draw lines
            if self.render:
                self.drawStaticObjects()

            # Sanity check
            if len(actions) != len(self.cars):
                print("Error: There must be action s for every car")
                exit(0)

            # Car loop
            for action, car in zip(actions, self.cars):
                # Apply action as first step
                if i % 10 == 0:
                    self.processAction(action, car)

                # Update cars
                self.tick(car)

            [self.move(ped) for ped in self.pedestrians]

            # Run simulation
            self.space.step(1 / self.timeStep)

            self.elapsed += 1
            allFinised = all([car.finished and not car.crashed for car in self.cars])
            if not self.allFinished:
                if allFinised:
                    self.allFinised = True
                    self.teamReward += self.maxTime-self.elapsed
                elif self.elapsed >= self.maxTime:
                    self.teamReward -= 2000

            if self.elapsed >= self.maxTime:
                finished = True

            # Get observations every 100 ms
            if i % 10 == 9:
                if self.observationType == ObservationType.Full:
                    observations.append([self.getFullState(car) for car in self.cars])
                else:
                    observations.append([self.getCarVision(car) for car in self.cars])

            # Render
            if self.render:
                pygame.display.flip()
                self.clock.tick(self.timeStep)
                cv2.waitKey(1)

        t2 = time.clock()
        #print((t2 - t1) * 1000)

        return self.getFullState(), observations, self.teamReward, self.carRewards, finished

    def drawStaticObjects(self):

        self.screen.fill((150, 150, 150))

        for road in self.roads:
            for line in road.Walkways:
                pygame.draw.line(self.screen,(255,255,0),line[0],line[1],1)


            for i,line in enumerate(road.Lanes):
                color = (255,255,255)
                thickness = 1
                if i == 0 or i == road.nLanes*2:
                    color = (255,0,0)
                elif i == road.nLanes:
                    thickness = 2
                pygame.draw.line(self.screen,color,line[0],line[1],thickness)

        self.space.debug_draw(self.draw_options)

    def getFullState(self,car=None):
        pass

    def getCarVision(self,car):
        pass

    def processAction(self,action,car):
        acc = action[0]
        steer = action[1]

        if acc != 0:
            car.accelerate(acc)
        if steer != 0:
            car.turn(steer)

    def tick(self,car):

        car.position = LanePosition.OffRoad
        index = self.cars.index(car)

        for road in self.roads:
            rPos = road.isPointOnRoad(car.shape.body.position,car.shape.body.angle)
            car.position = min(car.position,rPos)

        if (car.shape.body.position - car.goal).length < self.distThreshold:
            car.position = LanePosition.AtGoal
            car.finished = True
            car.shape.color = (255,255,255)
            self.carRewards[index] += 5000
        else:
            diff = (car.prevPos-car.goal).length - (car.shape.body.position-car.goal).length
            self.carRewards[index] += diff
            car.prevPos = car.shape.body.position
            if car.position == LanePosition.OffRoad:
                car.crash()
                self.carRewards[index] -= 2000
            elif car.position == LanePosition.InOpposingLane:
                self.carRewards[index] -= 10

    def move(self,pedestrian):
        if not pedestrian.dead:
            isOffRoad = self.isOffRoad(pedestrian.shape.body.position)
            isOut = self.isOut(pedestrian.shape.body.position)
            if pedestrian.moving > 0:
                pedestrian.moving = max(0,pedestrian.moving-self.timeDiff)
                if pedestrian.crossing:
                    if not pedestrian.beginCrossing and isOffRoad:
                        pedestrian.moving = 0
                        pedestrian.crossing = False
                        pedestrian.shape.body.velocity = pymunk.Vec2d(0,0)
                    elif pedestrian.beginCrossing and not isOffRoad:
                        pedestrian.beginCrossing = False
                if isOut:
                    pedestrian.moving = 0
                    pedestrian.shape.body.velocity = pymunk.Vec2d(0,0)
            else:
                if not pedestrian.crossing:
                    pedestrian.moving = random.randint(5000,30000)
                    speed = random.randint(-2,2)
                    dir = pedestrian.direction
                    if not isOffRoad:
                        pedestrian.crossing = True
                        pedestrian.beginCrossing = False
                        if speed == 0:
                            speed = 2
                    elif isOut:
                        dir = -pedestrian.direction if self.isOut(pedestrian.shape.body.position+pedestrian.direction) else pedestrian.direction
                    elif random.random() < 0.1:
                        pedestrian.crossing = True
                        pedestrian.beginCrossing = True
                        dir = pedestrian.normal if pedestrian.side else -pedestrian.normal
                        pedestrian.side = 0 if pedestrian.side else 1
                        speed = random.randint(1,2)
                    pedestrian.shape.body.velocity = pedestrian.speed*dir*speed
                else:
                    if isOffRoad:
                        pedestrian.crossing = False
                        pedestrian.beginCrossing = False


    def getUniqueSpots(self):

        roadSpots = np.array([10*road.nLanes for road in self.roads])
        roadSpotsId = np.array([sum(roadSpots[0:i+1]) for i in range(len(self.roads))])
        roadSpotsOffs = np.array([sum(roadSpots[0:i]) for i in range(len(self.roads))])
        nSpots = sum(roadSpots)

        spotIDs = np.random.permutation(nSpots)[:self.nPlayers]

        roadIDs = np.array([np.where(spotID < roadSpotsId)[0][0] for spotID in spotIDs])
        spotIDs -= roadSpotsOffs[roadIDs]

        laneIDs = spotIDs//5
        spotIDs = spotIDs%5

        return [self.roads[roadID].getSpot(laneID,spotID) for roadID,laneID,spotID in zip(roadIDs,laneIDs,spotIDs)]

    def createRandomPedestrians(self):

        roadIds = np.random.randint(0,len(self.roads),self.pedestrianNum)
        sideIds = np.random.randint(0,2,self.pedestrianNum)
        lenOffs = np.random.rand(self.pedestrianNum)
        widthOffs = np.random.rand(self.pedestrianNum)/2+0.25

        return [Pedestrian(self.roads[road].getWalkSpot(side,length,width),self.roads[road],side) for road, side, length, width in zip(roadIds,sideIds,lenOffs,widthOffs)]

    def createRandomObstacles(self):

        roadIds = np.random.randint(0,len(self.roads),self.obstacleNum)
        sideIds = np.random.randint(0,2,self.obstacleNum)
        lenOffs = np.random.rand(self.obstacleNum)
        widthOffs = np.random.rand(self.obstacleNum)/2+0.25

        obs = [Obstacle(self.roads[road].getWalkSpot(side,length,width),10,10) for road, side, length, width in zip(roadIds,sideIds,lenOffs,widthOffs)]

        return [ob for ob in obs if self.isOffRoad(ob.shape.body.position)]

    def isOffRoad(self,point):
        position = LanePosition.OffRoad

        for road in self.roads:
            rPos = road.isPointOnRoad(point, 0)
            position = min(position, rPos)

        return position == LanePosition.OffRoad

    def isOut(self,pos):
        return pos.x <= 0 or pos.y <= 0 or pos.x >= self.W or pos.y >= self.H

    def pedCollision(self,arbiter, space, data):
        return False

    def carCrash(self,arbiter, space, data):

        car1 = next(car for car in self.cars if (car.shape == arbiter.shapes[0]))
        car2 = next(car for car in self.cars if (car.shape == arbiter.shapes[1]))

        car1.crash()
        car2.crash()

        index1 = self.cars.index(car1)
        index2 = self.cars.index(car2)
        self.carRewards[index1] -= 2000
        self.carRewards[index2] -= 2000

        if car1.position != LanePosition.InRightLane:
            self.carRewards[index1] -= 2000

        if car2.position != LanePosition.InRightLane:
            self.carRewards[index2] -= 2000

        if car1.position == LanePosition.InRightLane and car2.position == LanePosition.InRightLane:
            v1 = arbiter.shapes[0].body.velocity
            v2 = arbiter.shapes[1].body.velocity
            p1 = car1.shape.body.position
            p2 = car2.shape.body.position
            dp = p1 - p2

            # Car is responsible if moving towards the other one
            if v1.length > 1 and math.cos(dp.angle - v1.angle) < -0.4:
                self.carRewards[index1] -= 2000

            if v2.length > 1 and math.cos(dp.angle - v2.angle) > 0.4:
                self.carRewards[index2] -= 2000

        return True

    def pedHit(self,arbiter, space, data):

        car = next(car for car in self.cars if (car.shape == arbiter.shapes[0]))
        ped = next(ped for ped in self.pedestrians if (ped.shape == arbiter.shapes[1]))

        ped.die()

        v1 = arbiter.shapes[0].body.velocity
        if v1.length > 1:
            car.crash()

            p1 = car.shape.body.position
            p2 = ped.shape.body.position
            dp = p1 - p2

            if math.cos(dp.angle - v1.angle) < -0.4:
                index = self.cars.index(car)
                self.carRewards[index] -= 5000

        return True

    def carHit(self,arbiter, space, data):

        car = next(car for car in self.cars if (car.shape == arbiter.shapes[0]))

        car.crash()

        index = self.cars.index(car)
        self.carRewards[index] -= 5000

        return True
