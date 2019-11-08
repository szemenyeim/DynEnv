from .Car import Car
from .Pedestrian import Pedestrian
from .Obstacle import Obstacle
from .Road import Road
from .utils import ObservationType, NoiseType
from .cutils import LanePosition
import cv2
import random
import time
import pymunk
import pymunk.pygame_util
import pygame

class DrivingEnvironment(object):

    def __init__(self,nPlayers,render=False,observationType = ObservationType.Partial,noiseType = NoiseType.Realistic, noiseMagnitude = 2):

        # Basic settings
        self.observationType = observationType
        self.noiseType = noiseType
        self.maxPlayers = 10
        self.nPlayers = min(nPlayers,self.maxPlayers)
        self.render = render

        # Noise
        self.noiseMagnitude = noiseMagnitude

        # Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.timeStep = 100.0
        self.distThreshold = 10

        # Setup scene
        self.W = 1750
        self.H = 1000

        # Setup roads
        self.roads = [
            Road(2,35,[pymunk.Vec2d(875,50),pymunk.Vec2d(875,950)]),
            Road(1,35,[pymunk.Vec2d(50,500),pymunk.Vec2d(1700,500)]),
        ]

        # Add cars
        self.cars = [Car(pymunk.Vec2d(55,515),random.randint(0,3),random.randint(0,1),pymunk.Vec2d(1000,540)) for i in range(self.nPlayers)]
        for car in self.cars:
            self.space.add(car.shape.body,car.shape)

        # Add random pedestrians

        # Add random obstacles

        # Setup ped-ped collision (ignore)

        # Setup car-car collision

        # Setup car-ped collision

        # Setup car-obst collision

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
        self.teamRewards = [0, 0]
        self.carRewards = [0,] * self.nPlayers
        observations = []
        finished = [False,] * self.nPlayers

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
                if i == 0:
                    self.processAction(action, car)

                # Update cars
                self.tick(car)

            # Run simulation
            self.space.step(1 / self.timeStep)

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
        print((t2 - t1) * 1000)

        return self.getFullState(), observations, self.teamRewards, self.carRewards, finished

    def drawStaticObjects(self):

        self.screen.fill((75, 75, 75))

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

        for road in self.roads:
            rPos = road.isPointOnRoad(car.shape.body.position,car.shape.body.angle)
            car.position = min(car.position,rPos)

        if (car.shape.body.position - car.goal).length < self.distThreshold:
            car.position = LanePosition.AtGoal