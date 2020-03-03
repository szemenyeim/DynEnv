import copy
import math
from .cutils import LanePosition, SightingType
import numpy as np


class Road:

    laneScaleFactor = 0.1

    def __init__(self,nLanes,width,points,hasWalkway = True):

        # Basic parameters
        self.nLanes = nLanes
        self.width = width
        self.points = points
        self.hasWalkway = hasWalkway
        self.width = width

        # Safe following distance (used to initialize car positions)
        self.followDist = 90

        # Direction and normal vectors
        self.direction = self.points[1]-self.points[0]
        self.length = self.direction.length
        self.direction = self.direction / self.length
        self.normal = copy.deepcopy(self.direction)
        self.normal.rotate(math.pi/2)

        # Setup lanes and walkways
        self.Lanes = [[self.points[0] + i*width*self.normal,self.points[1] + i*width*self.normal] for i in range(-nLanes,nLanes+1)]
        self.Walkways = [[self.points[0] + (nLanes+1)*width*self.normal,self.points[1] + (nLanes+1)*width*self.normal],
        [self.points[0] - (nLanes+1)*width*self.normal,self.points[1] - (nLanes+1)*width*self.normal] ]if hasWalkway else []

    # get distances from a Lane
    def getCarLaneDistances(self,carPos,carAngle):
        # Get difference
        pt = carPos - self.points[0]

        # Distance from middle
        dist = self.direction.cross(pt)/self.width
        if abs(dist) > 10:
            return np.array([[SightingType.NoSighting, 0, 0, 0, 0],])

        # Lane IDs
        laneTypes = np.array([1,] * self.nLanes + [-1,] * self.nLanes)

        # Angles
        a = self.direction.angle - carAngle
        c = math.cos(a)
        s = math.sin(a)
        distMult = 1

        # Flip distance and lanes
        if c >= 0:
            laneTypes *= -1
            c *= -1
            s *= -1
            distMult = -1

        # Angles for every lane
        c = np.array([c,]*self.nLanes*2)
        s = np.array([s,]*self.nLanes*2)

        # SightingTypes
        sighting = np.array([SightingType.Normal,]*self.nLanes*2)

        # get distances
        distances = ((dist + 0.5) + np.array([i for i in range(-self.nLanes,self.nLanes)]))*self.width*self.laneScaleFactor

        return np.stack([sighting,distances*distMult,c,s,laneTypes]).T

    # Decide if point ios on the road
    def isPointOnRoad(self,point,angle):

        # Get difference
        pt = point - self.points[0]

        # Distance from middle
        dist = self.direction.cross(pt)

        # Is on the road
        if abs(dist) >= self.nLanes*self.width+5:
            return LanePosition.OffRoad
        else:
            pos = LanePosition.OverRoad

            # Ged logituinal distance from starting point
            dirDist = self.direction.dot(pt)

            # If on the road
            if dirDist >= -10 and dirDist <= self.length+10:
                # Is on right side of the road
                relAngle = math.cos(self.direction.angle - angle)*dist
                pos = LanePosition.InRightLane if relAngle < 0 else LanePosition.InOpposingLane

            return pos

    # Determine location of a spot on a lane
    def getSpot(self,lane,spot):

        # Determine which end of the road we're on
        end = 1 if lane >= self.nLanes else 0
        pos = self.points[end]

        # Get spot and lane directions
        spotDir = (-self.direction if end else self.direction) * self.followDist
        laneDir = (self.normal if end else -self.normal) * self.width

        # Lane id
        lane = (lane-self.nLanes if end else lane)+0.5

        # Return position
        return pos + lane*laneDir + spot*spotDir,spotDir.angle

    # Get position for walkway spot
    def getWalkSpot(self,side,length,width):

        # Get longitudinal position
        center = self.Walkways[side][0] + length*(self.Walkways[side][1]-self.Walkways[side][0])

        # Width offset
        center += width*self.width*self.normal*(1 if side else -1)

        return center