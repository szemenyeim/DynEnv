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
        self.nPlayers = min(nPlayers*2,self.maxPlayers)
        self.render = render
        self.numTeams = 2

        if self.observationType == ObservationType.Image:

            print("Image type observation is not supported for this environment")
            exit(0)

        self.visId = 0#random.randint(0,self.nPlayers-1)

        # Setup scene
        self.W = 1750
        self.H = 1000

        # Normalization parameters
        self.normX = 1.0/self.W
        self.normY = 1.0/self.H
        self.normW = 1.0/7.5
        self.normH = 1.0/15

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
        self.maxVisDist = [(self.W * 0.4)**2, (self.W * 0.6)**2]

        # Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.timeStep = 100.0
        self.timeDiff = 10.0
        self.distThreshold = 10

        # Time rewards
        self.maxTime = 6000
        self.elapsed = 0
        self.allFinished = False

        # Setup roads
        self.roads = [
            Road(2,35,[pymunk.Vec2d(875,0),pymunk.Vec2d(875,1000)]),
            Road(1,35,[pymunk.Vec2d(0,500),pymunk.Vec2d(1750,500)]),
        ]
        self.laneNum = len(self.roads) + sum([r.nLanes*2 for r in self.roads])

        self.buildings = [
            Obstacle(pymunk.Vec2d(365.0,200.0),400,225),
            Obstacle(pymunk.Vec2d(365.0,800.0),400,225),
            Obstacle(pymunk.Vec2d(1385.0,200.0),400,225),
            Obstacle(pymunk.Vec2d(1385.0,800.0),400,225),
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
        self.pedestrianNum = random.randint(15,30)
        self.pedestrians = self.createRandomPedestrians()
        for ped in self.pedestrians:
            self.space.add(ped.shape.body,ped.shape)

        # Add random obstacles
        self.obstacleNum = random.randint(10,20)
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

    # Reset env
    def reset(self):
        self.__init__(self.nPlayers,self.render,self.observationType,self.noiseType,self.noiseMagnitude)
        observations = []
        if self.observationType == ObservationType.Full:
            observations.append([self.getFullState(car) for car in self.cars])
        else:
            observations.append([self.getCarVision(car) for car in self.cars])
        return observations

    # Set random seed
    def setRandomSeed(self,seed):
        np.random.seed(seed)
        random.seed(seed)
        return self.reset()

    # Observation space
    def getObservationSize(self):
        if self.observationType == ObservationType.Full:
            return [1,self.nPlayers,[[7,],[self.nPlayers-1,5],[self.obstacleNum,4],[self.pedestrianNum,2],[self.laneNum,5]]]
        else:
            return [1,self.nPlayers, 5, [7,5,5,2,5]]

    # Action space
    def getActionSize(self):
        return [self.nPlayers,[
            ['cont',2,[0,0],[6,6]],
        ]]

    def step(self,actions):
        t1 = time.clock()

        # Setup reward and state variables
        self.teamReward = 0.0
        self.carRewards = np.array([0.0,] * self.nPlayers)
        finished = False
        observations = []

        # Run simulation for 500 ms (time for every action)
        for i in range(10):

            # Draw lines
            if self.render:
                self.drawStaticObjects()

            # Sanity check
            if actions.shape != (len(self.cars),2):
                raise Exception("Error: There must be 2 actions for every car")

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


        self.carRewards += self.teamReward

        t2 = time.clock()
        #print((t2 - t1) * 1000)
        if finished:
            print("Episode finished")

        return self.getFullState(), observations, self.carRewards, finished

    def drawStaticObjects(self):

        # Gray screen
        self.screen.fill((150, 150, 150))

        # Draw roads
        for road in self.roads:

            # Walkway edges
            for line in road.Walkways:
                pygame.draw.line(self.screen,(255,255,0),line[0],line[1],1)

            # Draw lanes
            for i,line in enumerate(road.Lanes):

                # Default params
                color = (255,255,255)
                thickness = 1

                # Edges with red
                if i == 0 or i == road.nLanes*2:
                    color = (255,0,0)
                # Middle double thick
                elif i == road.nLanes:
                    thickness = 2

                # Draw
                pygame.draw.line(self.screen,color,line[0],line[1],thickness)

        # draw everything else
        self.space.debug_draw(self.draw_options)

    # Process actions
    def processAction(self,action,car):

        # Get actions
        acc = action[0]
        steer = action[1]

        # Sanity checks
        if np.abs(acc) > 3:
            raise Exception("Error: Acceleration must be between +/-3")
        if np.abs(steer) > 3:
            raise Exception("Error: Steering must be between +/-3")

        # Apply actions to car
        if acc != 0:
            car.accelerate(acc.item())
        if steer != 0:
            car.turn(steer.item())

    # Update car status
    def tick(self,car):

        # Get params
        car.position = LanePosition.OffRoad
        index = self.cars.index(car)

        # Get car position relative to roads
        for road in self.roads:
            rPos = road.isPointOnRoad(car.getPos(),car.getAngle())
            car.position = min(car.position,rPos)

        # If reached goal finish and add reward
        if (car.getPos() - car.goal).length < self.distThreshold:
            car.position = LanePosition.AtGoal
            car.finished = True
            car.shape.color = (255,255,255)
            self.carRewards[index] += 5000
        else:
            # REward for getting closer
            diff = (car.prevPos-car.goal).length - (car.getPos()-car.goal).length
            self.carRewards[index] += diff

            # Update previous position
            car.prevPos = car.getPos()

            # crash for leaving road
            if car.position == LanePosition.OffRoad:
                car.crash()
                self.carRewards[index] -= 2000
            # Add small punichment for being in opposite lane
            elif car.position == LanePosition.InOpposingLane:
                self.carRewards[index] -= 10

    # Update function for pedestrians
    def move(self,pedestrian):

        # Dead pedestrians don't move
        if not pedestrian.dead:

            # Get pedestrian status
            isOffRoad = self.isOffRoad(pedestrian.getPos())
            isOut = self.isOut(pedestrian.getPos())

            # If there is still time left for the current movement
            if pedestrian.moving > 0:

                # Decrease timer
                pedestrian.moving = max(0,pedestrian.moving-self.timeDiff)

                # If the pedestrian is crossing the road
                if pedestrian.crossing:

                    # If the pedestrian finished crossing
                    if not pedestrian.beginCrossing and isOffRoad:
                        # Reset everything and stop
                        pedestrian.moving = 0
                        pedestrian.crossing = False
                        pedestrian.shape.body.velocity = pymunk.Vec2d(0,0)

                    # If the pedestrian just started crossing and got on the road
                    elif pedestrian.beginCrossing and not isOffRoad:
                        # Set begin to false
                        pedestrian.beginCrossing = False
                # If pedestrian walked out, stop and get new direction in next tick
                if isOut:
                    pedestrian.moving = 0
                    pedestrian.shape.body.velocity = pymunk.Vec2d(0,0)
            # If movement expired
            else:
                # If the pedestrian is not crossing, get new movement
                if not pedestrian.crossing:

                    # Get random time and speed
                    pedestrian.moving = random.randint(5000,30000)
                    speed = random.randint(-2,2)
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
                        dir = -pedestrian.direction if self.isOut(pedestrian.getPos()+pedestrian.direction) else pedestrian.direction
                    # Otherwise cross the road with 10% chance
                    elif random.random() < 0.1:

                        # Setup normal crossing
                        pedestrian.crossing = True
                        pedestrian.beginCrossing = True

                        # Get direction
                        dir = pedestrian.normal if pedestrian.side else -pedestrian.normal

                        # Change side
                        pedestrian.side = 0 if pedestrian.side else 1

                        # Set non-zero speed
                        speed = random.randint(1,2)

                    # Set new speed
                    pedestrian.shape.body.velocity = pedestrian.speed*dir*speed
                # If pedestrian was crossing but now is off the road, reset crossing vars
                elif isOffRoad:
                        pedestrian.crossing = False
                        pedestrian.beginCrossing = False

    # Is point off the road
    def isOffRoad(self,point):

        # Default
        position = LanePosition.OffRoad

        # Get position relative to the roads
        for road in self.roads:
            rPos = road.isPointOnRoad(point, 0)
            position = min(position, rPos)

        # Return
        return position == LanePosition.OffRoad

    # Is point out of the field
    def isOut(self,pos):
        return pos.x <= 0 or pos.y <= 0 or pos.x >= self.W or pos.y >= self.H

    # Get unique sports on the road
    def getUniqueSpots(self):

        # 5 spots for every lane (road.nLanes is the lanes in a single direction)
        roadSpots = np.array([10*road.nLanes for road in self.roads])

        # IDs and offsets for indexing
        roadSpotsIDs = np.array([sum(roadSpots[0:i+1]) for i in range(len(self.roads))])
        roadSpotsOffs = np.array([sum(roadSpots[0:i]) for i in range(len(self.roads))])

        # Get random spots
        nSpots = sum(roadSpots)
        spotIDs = np.random.permutation(nSpots)[:self.nPlayers]

        # Get the ID of the road for each spot
        roadIDs = np.array([np.where(spotID < roadSpotsIDs)[0][0] for spotID in spotIDs])

        # Subtract road offset
        spotIDs -= roadSpotsOffs[roadIDs]

        # Get index of the lane and the spot
        laneIDs = spotIDs//5
        spotIDs = spotIDs%5

        # Get the actual spots
        return [self.roads[roadID].getSpot(laneID,spotID) for roadID,laneID,spotID in zip(roadIDs,laneIDs,spotIDs)]

    # Create random pedestrians
    def createRandomPedestrians(self):

        # Select roads, sides
        roadIds = np.random.randint(0,len(self.roads),self.pedestrianNum)
        sideIds = np.random.randint(0,2,self.pedestrianNum)

        # Create random placements along the length and width of the walkways
        lenOffs = np.random.rand(self.pedestrianNum)
        widthOffs = np.random.rand(self.pedestrianNum)/2+0.25

        # Get actual pedestrians
        return [Pedestrian(self.roads[road].getWalkSpot(side,length,width),self.roads[road],side) for road, side, length, width in zip(roadIds,sideIds,lenOffs,widthOffs)]

    # Create random obstacles
    def createRandomObstacles(self):

        # Select roads, sides
        roadIds = np.random.randint(0,len(self.roads),self.obstacleNum)
        sideIds = np.random.randint(0,2,self.obstacleNum)

        # Create random placements along the length and width of the walkways
        lenOffs = np.random.rand(self.obstacleNum)
        widthOffs = np.random.rand(self.obstacleNum)/2+0.25

        # Get actual obstacles
        obs = [Obstacle(self.roads[road].getWalkSpot(side,length,width),10,10) for road, side, length, width in zip(roadIds,sideIds,lenOffs,widthOffs)]

        # Remove the ones on the roads
        return [ob for ob in obs if self.isOffRoad(ob.getPos())]

    # Don't handle pedestrian obstacle collision
    def pedCollision(self,arbiter, space, data):
        return False

    # Car collision
    def carCrash(self,arbiter, space, data):

        # Get cars
        car1 = next(car for car in self.cars if (car.shape == arbiter.shapes[0]))
        car2 = next(car for car in self.cars if (car.shape == arbiter.shapes[1]))

        # Crash them
        car1.crash()
        car2.crash()

        # Punish
        index1 = self.cars.index(car1)
        index2 = self.cars.index(car2)
        self.carRewards[index1] -= 2000
        self.carRewards[index2] -= 2000

        # Punish car in the wrong lane extra
        if car1.position != LanePosition.InRightLane:
            self.carRewards[index1] -= 2000

        if car2.position != LanePosition.InRightLane:
            self.carRewards[index2] -= 2000

        # If both in the rights lane
        if car1.position == LanePosition.InRightLane and car2.position == LanePosition.InRightLane:

            # Get velocities and relative position
            v1 = arbiter.shapes[0].body.velocity
            v2 = arbiter.shapes[1].body.velocity
            p1 = car1.getPos()
            p2 = car2.getPos()
            dp = p1 - p2

            # Car is responsible if moving towards the other one
            if v1.length > 1 and math.cos(angle(dp) - angle(v1)) < -0.4:
                self.carRewards[index1] -= 2000

            if v2.length > 1 and math.cos(angle(dp) - angle(v2)) > 0.4:
                self.carRewards[index2] -= 2000

        return True

    # Handle car hitting pedestrian
    def pedHit(self,arbiter, space, data):

        # Get objects
        car = next(car for car in self.cars if (car.shape == arbiter.shapes[0]))
        ped = next(ped for ped in self.pedestrians if (ped.shape == arbiter.shapes[1]))

        # Blergh
        ped.die()

        # Get velocity
        v1 = arbiter.shapes[0].body.velocity
        if v1.length > 1:

            # Get relative positions
            p1 = car.getPos()
            p2 = ped.getPos()
            dp = p1 - p2

            # Crash car if it actually hit pedestrian
            if math.cos(angle(dp) - angle(v1)) < -0.4:
                car.crash()
                index = self.cars.index(car)
                self.carRewards[index] -= 5000

        return True

    # Car-obstacle crash
    def carHit(self,arbiter, space, data):

        # Get car
        car = next(car for car in self.cars if (car.shape == arbiter.shapes[0]))

        # crash car
        car.crash()

        # Punish
        index = self.cars.index(car)
        self.carRewards[index] -= 5000

        return True

    # Gett correct state
    def getFullState(self,car=None):

        # Get lanes
        lanes = []
        for l in self.roads:
            lanes += [[normalize(l.Lanes[i - l.nLanes][0].x, self.normX), normalize(l.Lanes[i - l.nLanes][0].y, self.normY),
                       normalize(l.Lanes[i - l.nLanes][1].x, self.normX), normalize(l.Lanes[i - l.nLanes][1].y, self.normY),
                       (1 if abs(i) == l.nLanes else (-1 if i == 0 else 0))] for i in range(-l.nLanes, l.nLanes + 1)]

        # If complete state
        if car is None:

            # Just add cars
            state = [
                np.array([[normalize(c.getPos().x,self.normX),normalize(c.getPos().y,self.normY), c.getAngle(),
                           normalize(c.width, self.normW), normalize(c.height, self.normH)] for c in self.cars]),
            ]
        # Otherwise add self observation separately
        else:
            state = [
                np.array([normalize(car.getPos().x,self.normX),normalize(car.getPos().y,self.normY), car.getAngle(),
                          normalize(car.width, self.normW), normalize(car.height, self.normH),
                          normalize(car.goal.x,self.normX),normalize(car.goal.y,self.normY)]),
                np.array([[normalize(c.getPos().x,self.normX),normalize(c.getPos().y,self.normY), c.getAngle(),
                           normalize(c.width, self.normW), normalize(c.height, self.normH)] for c in self.cars if c != car]),
            ]

        # Add obstacles, pedestrians and lanes
        state += [
            np.array([[normalize(o.getPos().x, self.normX), normalize(o.getPos().y, self.normY),
                       normalize(o.width, self.normW), normalize(o.height, self.normH)] for o in self.obstacles]),
            np.array([[normalize(p.getPos().x, self.normX), normalize(p.getPos().y, self.normY)] for p in self.pedestrians]),
            np.array(lanes)
        ]

        return state

    # Get car observation
    def getCarVision(self,car):

        # Get detections within radius
        selfDet = [SightingType.Normal, car.getPos(), car.getAngle(), car.getPoints(), car.width, car.height, car.goal]
        carDets = [isSeenInRadius(c.getPos(),c.getPoints(),c.getAngle(),selfDet[1],selfDet[2],self.maxVisDist[0],self.maxVisDist[1]) + [c.width, c.height] for c in self.cars if c != car]
        obsDets = [isSeenInRadius(o.getPos(),o.points,0,selfDet[1],selfDet[2],self.maxVisDist[0],self.maxVisDist[1]) + [o.width, o.height] for o in self.obstacles]
        buildDets = [isSeenInRadius(b.getPos(),b.points,0,selfDet[1],selfDet[2],20000000,20000000) for b in self.buildings] # Buildings are always seen
        pedDets = [isSeenInRadius(p.getPos(),None,0,selfDet[1],selfDet[2],self.maxVisDist[0],self.maxVisDist[1]) for p in self.pedestrians]
        laneDets = []
        for l in self.roads:
            laneDets += [getLineInRadius(l.Lanes[i+l.nLanes],selfDet[1],selfDet[2],self.maxVisDist[1])
                         + [(1 if abs(i) == l.nLanes else (-1 if i == 0 else 0)), ]
                        for i in range(-l.nLanes,l.nLanes+1)]

        # Remove objects not seen (this is to reduce computation down the road)
        carDets = [c for i,c in enumerate(carDets) if c[0] != SightingType.NoSighting]
        pedDets = [ped for i,ped in enumerate(pedDets) if ped[0] != SightingType.NoSighting]
        obsDets = [obs for i,obs in enumerate(obsDets) if obs[0] != SightingType.NoSighting]
        laneDets = [lane for i,lane in enumerate(laneDets) if lane[0] != SightingType.NoSighting]

        # Get objects occluded by buildings
        buildCarInter = [max([doesInteractPoly(c,b,0) for b in buildDets]) for c in carDets]
        buildPedInter = [max([doesInteractPoly(p,b,0) for b in buildDets]) for p in pedDets]
        buildObsInter = [max([doesInteractPoly(o,b,0) for b in buildDets]) for o in obsDets]

        # Remove occluded objects (this is to reduce computation down the road)
        carDets = [c for i,c in enumerate(carDets) if buildCarInter[i] != InteractionType.Occlude]
        pedDets = [ped for i,ped in enumerate(pedDets) if buildPedInter[i] != InteractionType.Occlude]
        obsDets = [obs for i,obs in enumerate(obsDets) if buildObsInter[i] != InteractionType.Occlude]

        # Get pedestrian-car and pedestrian-obstacle interactions
        carPedInter = [max([doesInteractPoly(p,c,400) for c in carDets]) for p in pedDets] if carDets else [InteractionType.NoInter,]*len(pedDets)
        obsPedInter = [max([doesInteractPoly(p,o,400) for o in obsDets]) for p in pedDets] if obsDets else [InteractionType.NoInter,]*len(pedDets)
        pedInter = max(carPedInter,obsPedInter)

        # set occluded pedestrians to NoSighting
        [filterOcclude(p,pedInter[i]) for i,p in enumerate(pedDets)]

        # Add noise: Self, Car, Obs, Ped and Lane
        addNoiseRect(selfDet,  self.noiseType, InteractionType.NoInter, self.noiseMagnitude, self.randBase, self.maxVisDist[1])
        [addNoiseRect(c, self.noiseType, InteractionType.NoInter, self.noiseMagnitude, self.randBase, self.maxVisDist[1], True) for i,c in enumerate(carDets)]
        [addNoiseRect(ped, self.noiseType, pedInter[i], self.noiseMagnitude, self.randBase, self.maxVisDist[0]) for i,ped in enumerate(pedDets)]
        [addNoiseRect(obs, self.noiseType, InteractionType.NoInter, self.noiseMagnitude, self.randBase, self.maxVisDist[1], True) for i,obs in enumerate(obsDets)]
        [addNoiseLine(lane, self.noiseType, self.noiseMagnitude, self.randBase, self.maxVisDist[1]) for i,lane in enumerate(laneDets)]

        # Cars and obstacles might by misclassified - move them in the other list
        for c in carDets:
            if c[0] == SightingType.Misclassified:
                obsDets.append((SightingType.Normal,c[1],c[2],c[3],c[4],c[5]))
        for obs in obsDets:
            if obs[0] == SightingType.Misclassified:
                carDets.append((SightingType.Normal,obs[1],obs[2],obs[3],obs[4],obs[5]))

        # Random false positives
        for i in range(10):
            if random.random() < self.randBase:

                # Class
                c = random.randint(0,5)

                # Distance and angle
                d = random.random()*self.maxVisDist[1]
                a1 = random.random()*2*math.pi
                pos = pymunk.Vec2d(d,0)
                pos.rotate(a1)

                # Object angle
                angle = random.random()*2*math.pi

                # car or obstacle
                if c <= 1:

                    # Width and height
                    w = random.random()*5+5
                    h = random.random()*10+5

                    # create corners
                    obs = [Vec2d(h, w), Vec2d(-h, w), -Vec2d(h, w), Vec2d(h, -w)]
                    [ob.rotate(angle) for ob in obs]
                    obs = [ob + pos for ob in obs]

                    # Add objects
                    if c == 0:
                        carDets.insert(len(carDets),
                                   [SightingType.Normal,pos,angle,obs,w,h])
                    else:
                        obsDets.insert(len(obsDets),
                                   [SightingType.Normal,pos,angle,obs,w,h])
                # Add pedestrian
                elif c == 2:
                    pedDets.insert(len(pedDets),
                                   [SightingType.Normal,pos])
                # Add lane
                elif c == 3:
                    # Get second point
                    a2 = random.random()*2*math.pi
                    pos2 = pymunk.Vec2d(d,0)
                    pos2.rotate(a2)

                    # Add lane
                    laneDets.insert(len(laneDets),
                                   [SightingType.Normal,pos,pos2,random.randint(-1,1)])

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

        if self.render and self.cars.index(car) == self.visId:

            # Visualization image size
            H = self.H//2
            W = self.W//2
            img = np.zeros((H*2,W*2,3)).astype('uint8')

            # Draw all objects
            # Partially seen and distant objects are dim
            # Objects are drawn from the robot center

            # Draw buildings first
            for build in buildDets:
                color = (255,255,255)
                points = build[3]
                cv2.fillConvexPoly(img,np.array([(int(p.x+W),int(-p.y+H)) for p in points]),color)

            # draw self
            color = (255,255,0)
            points = [p - car.getPos() for p in selfDet[3]]
            cv2.fillConvexPoly(img,np.array([(int(p.x+W),int(-p.y+H)) for p in points]),color)

            # draw lane (color based on type)
            for lane in laneDets:
                color = (0,0,255) if lane[3] == 1 else ((0,255,0) if lane[3] == -1 else (255,255,255))
                if lane[0] != SightingType.Normal:
                    color = (color[0]//2,color[1]//2,color[2]//2)
                cv2.line(img,(int(W+lane[1].x),int(-lane[1].y+H)),(int(W+lane[2].x),int(-lane[2].y+H)),color,2)

            # draw cars
            for c in carDets:
                color = (0,255,0) if c[0] == SightingType.Normal else (0,127,0)
                points = c[3]
                cv2.fillConvexPoly(img,np.array([(int(p.x+W),int(-p.y+H)) for p in points]),color)

            # draw obstacles
            for obs in obsDets:
                color = (255,0,255) if obs[0] == SightingType.Normal else (127,0,127)
                points = obs[3]
                cv2.fillConvexPoly(img,np.array([(int(p.x+W),int(-p.y+H)) for p in points]),color)

            # draw pedestrians
            for ped in pedDets:
                color = (255,0,0) if ped[0] == SightingType.Normal else (127,0,0)
                point = ped[1]
                cv2.circle(img,(int(point.x+W),int(-point.y+H)),5,color,-1)

            cv2.imshow(("Car %d" % self.cars.index(car)),img)

        # Convert to numpy
        selfDet = np.array([[normalize(selfDet[1].x,self.normX),normalize(selfDet[1].y,self.normY),selfDet[2],
                            normalize(selfDet[4],self.normW),normalize(selfDet[5],self.normH),
                            normalize(selfDet[6].x,self.normX),normalize(selfDet[6].y,self.normY)],])
        carDets = np.array([[normalize(car[1].x,self.normX),normalize(car[1].y,self.normY),car[2],
                             normalize(car[4],self.normW),normalize(car[5],self.normH)] for car in carDets])
        obsDets = np.array([[normalize(obs[1].x,self.normX),normalize(obs[1].y,self.normY),obs[2],
                             normalize(obs[4],self.normW),normalize(obs[5],self.normH)] for obs in obsDets])
        pedDets = np.array([[normalize(ped[1].x,self.normX),normalize(ped[1].y,self.normY)] for ped in pedDets])
        laneDets = np.array([[normalize(lane[1].x,self.normX),normalize(lane[1].y,self.normY),
                              normalize(lane[2].x,self.normX),normalize(lane[2].y,self.normY), lane[3]] for lane in laneDets])

        # return
        return selfDet,carDets,obsDets,pedDets,laneDets

    # For compatibility
    def close(self):
        pass

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
