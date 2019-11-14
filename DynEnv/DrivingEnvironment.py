# coding=utf-8
from .Car import Car
from .Pedestrian import Pedestrian
from .Obstacle import Obstacle
from .Road import Road
from .utils import ObservationType, NoiseType, CollisionType
from .cutils import *
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

        self.visId = 0#random.randint(0,self.nPlayers-1)

        # Setup scene
        self.W = 1750
        self.H = 1000

        # Noise
        # Vision settings
        if noiseMagnitude < 0 or noiseMagnitude > 5:
            print("Error: The noise magnitude must be between 0 and 5!")
            exit(0)
        if observationType == ObservationType.Full and noiseMagnitude > 0:
            print(
                "Warning: Full observation type does not support noisy observations, but your noise magnitude is set to a non-zero value! (The noise setting has no effect in this case)")
        self.randBase = 0.01 * noiseMagnitude
        self.noiseMagnitude = noiseMagnitude
        self.maxVisDist = [self.W * 0.4, self.W * 0.6]

        # Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.timeStep = 100.0
        self.timeDiff = 10.0
        self.distThreshold = 10

        # Time rewards
        self.maxTime = 2000
        self.elapsed = 0
        self.allFinished = False

        # Setup roads
        self.roads = [
            Road(2,35,[pymunk.Vec2d(875,0),pymunk.Vec2d(875,1000)]),
            Road(1,35,[pymunk.Vec2d(0,500),pymunk.Vec2d(1750,500)]),
        ]

        self.buildings = [
            Obstacle(pymunk.Vec2d(365,200),400,225),
            Obstacle(pymunk.Vec2d(365,800),400,225),
            Obstacle(pymunk.Vec2d(1385,200),400,225),
            Obstacle(pymunk.Vec2d(1385,800),400,225),
        ]
        for building in self.buildings:
            self.space.add(building.shape.body,building.shape)

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
        self.pedestrianNum = 20
        self.pedestrians = self.createRandomPedestrians()
        for ped in self.pedestrians:
            self.space.add(ped.shape.body,ped.shape)

        # Add random obstacles
        self.obstacleNum = 10
        self.obstacles = self.createRandomObstacles()
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
            rPos = road.isPointOnRoad(car.getPos(),car.getAngle())
            car.position = min(car.position,rPos)

        if (car.getPos() - car.goal).length < self.distThreshold:
            car.position = LanePosition.AtGoal
            car.finished = True
            car.shape.color = (255,255,255)
            self.carRewards[index] += 5000
        else:
            diff = (car.prevPos-car.goal).length - (car.getPos()-car.goal).length
            self.carRewards[index] += diff
            car.prevPos = car.getPos()
            if car.position == LanePosition.OffRoad:
                car.crash()
                self.carRewards[index] -= 2000
            elif car.position == LanePosition.InOpposingLane:
                self.carRewards[index] -= 10

    def move(self,pedestrian):
        if not pedestrian.dead:
            isOffRoad = self.isOffRoad(pedestrian.getPos())
            isOut = self.isOut(pedestrian.getPos())
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
                        dir = -pedestrian.direction if self.isOut(pedestrian.getPos()+pedestrian.direction) else pedestrian.direction
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

        return [ob for ob in obs if self.isOffRoad(ob.getPos())]

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
            p1 = car1.getPos()
            p2 = car2.getPos()
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

            p1 = car.getPos()
            p2 = ped.getPos()
            dp = p1 - p2

            if math.cos(dp.angle - v1.angle) < -0.4:
                car.crash()
                index = self.cars.index(car)
                self.carRewards[index] -= 5000

        return True

    def carHit(self,arbiter, space, data):

        car = next(car for car in self.cars if (car.shape == arbiter.shapes[0]))

        car.crash()

        index = self.cars.index(car)
        self.carRewards[index] -= 5000

        return True

    def getFullState(self,car=None):
        lanes = []
        for l in self.roads:
            lanes += [[l.Lanes[i-l.nLanes][0], l.Lanes[i-l.nLanes][1],
                       (1 if abs(i) == l.nLanes else (2 if i == 0 else 0))] for i in range(-l.nLanes, l.nLanes + 1)]
        if car is None:
            state = [
                [[c.getPos(), c.points, c.getAngle()] for c in self.cars] +
                [[o.getPos(), o.points] for o in self.obstacles] +
                [[p.getPos(),] for p in self.pedestrians] +
                lanes
            ]
        else:
            state = [
                [[car.getPos(), car.getAngle()]] +
                [[c.getPos(), c.getAngle()] for c in self.cars if c != car] +
                [[o.getPos(), ] for o in self.obstacles] +
                [[p.getPos(), ] for p in self.pedestrians] +
                lanes
            ]

        return state

    def getCarVision(self,car):

        selfDet = [SightingType.Normal, car.getPos(), car.getPoints(), car.getAngle(), car.goal]

        carDets = [isSeenInRadius(c.getPos(),c.getPoints(),c.getAngle(),selfDet[1],selfDet[3],self.maxVisDist[1]) for c in self.cars if c != car]
        obsDets = [isSeenInRadius(o.getPos(),o.points,0,selfDet[1],selfDet[3],self.maxVisDist[1]) for o in self.obstacles]
        buildDets = [isSeenInRadius(b.getPos(),b.points,0,selfDet[1],selfDet[3],2000) for b in self.buildings]
        pedDets = [isSeenInRadius(p.getPos(),[],0,selfDet[1],selfDet[3],self.maxVisDist[0]) for p in self.pedestrians]
        laneDets = []
        for l in self.roads:
            laneDets += [getLineInRadius(l.Lanes[i+l.nLanes],selfDet[1],selfDet[3],self.maxVisDist[1])
                         + [(1 if abs(i) == l.nLanes else (2 if i == 0 else 0)), ]
                        for i in range(-l.nLanes,l.nLanes+1)]

        buildCarInter = [max([doesInteractPoly(c,b,0) for b in buildDets]) for c in carDets]
        buildPedInter = [max([doesInteractPoly(p,b,0) for b in buildDets]) for p in pedDets]
        buildObsInter = [max([doesInteractPoly(o,b,0) for b in buildDets]) for o in obsDets]
        carPedInter = [max([doesInteractPoly(p,c,0) for c in carDets]) for p in pedDets] if carDets else [InteractionType.NoInter,]*len(pedDets)
        obsPedInter = [max([doesInteractPoly(p,o,0) for o in obsDets]) for p in pedDets]
        pedInter = max(buildPedInter,carPedInter,obsPedInter)

        # Add noise: Car, Obs, Ped
        addNoiseRect(selfDet,  self.noiseType, InteractionType.NoInter, self.noiseMagnitude, self.randBase, self.maxVisDist[1])
        [addNoiseRect(c, self.noiseType, buildCarInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[1], True) for i,c in enumerate(carDets)]
        [addNoiseRect(ped, self.noiseType, pedInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[0]) for i,ped in enumerate(pedDets)]
        [addNoiseRect(obs, self.noiseType, buildObsInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[1], True) for i,obs in enumerate(obsDets)]
        [addNoiseLine(lane, self.noiseType, self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i,lane in enumerate(laneDets)]

        # Cars and obstacles might by misclassified - move them in the other list
        for c in carDets:
            if c[0] == SightingType.Misclassified:
                obsDets.append((SightingType.Normal,c[1],c[2],c[3]))
        for obs in obsDets:
            if obs[0] == SightingType.Misclassified:
                carDets.append((SightingType.Normal,obs[1],obs[2],obs[3]))

        # Random false positives
        for i in range(10):
            if random.random() < self.randBase:
                c = random.randint(0,5)
                d = random.random()*self.maxVisDist[1]
                a1 = random.random()*2*math.pi
                angle = random.random()*2*math.pi
                pos = pymunk.Vec2d(d,0)
                pos.rotate(a1)

                if c <= 1:
                    w = random.random()*5+5
                    h = random.random()*10+5
                    obs = [Vec2d(h, w), Vec2d(-h, w), -Vec2d(h, w), Vec2d(h, -w)]
                    [ob.rotate(angle) for ob in obs]
                    obs = [ob + pos for ob in obs]
                    if c == 0:
                        carDets.insert(len(carDets),
                                   [SightingType.Normal,pos,obs,angle])
                    else:
                        obsDets.insert(len(obsDets),
                                   [SightingType.Normal,pos,obs,angle])
                elif c == 2:
                    pedDets.insert(len(pedDets),
                                   [SightingType.Normal,pos,[],0])
                elif c == 3:
                    a2 = random.random()*2*math.pi
                    pos2 = pymunk.Vec2d(d,0)
                    pos2.rotate(a2)
                    laneDets.insert(len(laneDets),
                                   [SightingType.Normal,pos,pos2,random.randint(0,2)])

        # FP Pedestrians near cars and obstacles
        if self.noiseType == NoiseType.Realistic:
            for c in carDets:
                if c[0] == SightingType.Normal and random.random() < self.randBase*10 and c[1].length < 250:
                    offset = pymunk.Vec2d(2*random.random()-1.0,2*random.random()-1.0)*10
                    pedDets.insert(len(pedDets),
                                   [SightingType.Normal,c[1]+offset,[],0])

        # Remove occlusion and misclassified originals
        carDets = [c for i,c in enumerate(carDets) if c[0] != SightingType.NoSighting and c[0] != SightingType.Misclassified]
        pedDets = [ped for i,ped in enumerate(pedDets) if ped[0] != SightingType.NoSighting]
        obsDets = [obs for i,obs in enumerate(obsDets) if obs[0] != SightingType.NoSighting and obs[0] != SightingType.Misclassified]
        laneDets = [lane for i,lane in enumerate(laneDets) if lane[0] != SightingType.NoSighting]

        if self.observationType == ObservationType.Image:

            print("Image type observation is not supported for this environment")
            exit(0)

        if self.render and self.cars.index(car) == self.visId:

            # Visualization image size
            H = self.H//2
            W = self.W//2
            img = np.zeros((H*2,W*2,3)).astype('uint8')

            # Draw all objects
            # Partially seen and distant objects are dim
            # Objects are drawn from the robot center

            for build in buildDets:
                color = (255,255,255)
                points = build[2]
                cv2.fillConvexPoly(img,np.array([(int(p.x+W),int(-p.y+H)) for p in points]),color)

            color = (255,255,0)
            points = [p - car.getPos() for p in selfDet[2]]
            cv2.fillConvexPoly(img,np.array([(int(p.x+W),int(-p.y+H)) for p in points]),color)

            for lane in laneDets:
                color = (0,0,255) if lane[3] == 1 else ((0,255,0) if lane[3] == 2 else (255,255,255))
                if lane[0] != SightingType.Normal:
                    color = (color[0]//2,color[1]//2,color[2]//2)
                cv2.line(img,(int(W+lane[1].x),int(-lane[1].y+H)),(int(W+lane[2].x),int(-lane[2].y+H)),color,2)

            for c in carDets:
                color = (0,255,0) if c[0] == SightingType.Normal else (0,127,0)
                points = c[2]
                cv2.fillConvexPoly(img,np.array([(int(p.x+W),int(-p.y+H)) for p in points]),color)

            for ped in pedDets:
                color = (255,0,0) if ped[0] == SightingType.Normal else (127,0,0)
                point = ped[1]
                cv2.circle(img,(int(point.x+W),int(-point.y+H)),5,color,-1)

            for obs in obsDets:
                color = (255,0,255) if obs[0] == SightingType.Normal else (127,0,127)
                points = obs[2]
                cv2.fillConvexPoly(img,np.array([(int(p.x+W),int(-p.y+H)) for p in points]),color)

            cv2.imshow(("Car %d" % self.cars.index(car)),img)

        return selfDet,carDets,buildDets,obsDets,pedDets,laneDets

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
               "        Observations [position, corners]\n" \
               "        Pedestrians [position]\n" \
               "        Lanes [point1, point2, type]\n" \
               "    Observations: Contains car observations (in the same order as the cars are in the full state):\n" \
               "        Self detection: [sightingType, position, corners, angle, goal]\n" \
               "        Car detections: [sightingType, position, corners, angle]\n" \
               "        Building detections: [sightingType, position, corners]\n" \
               "        Obstacle detections: [sightingType, position, corners]\n" \
               "        Pedestrian detections: [sightingType, position]\n" \
               "        Lane detections: [sightingType, point1, point2, type]\n" \
               "    Team rewards: rewards for each team\n" \
               "    Car rewards: rewards for each car (in the same order as in state)\n" \
               "    Finished: Game over flag"
