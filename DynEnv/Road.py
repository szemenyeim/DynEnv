import copy
import math
from pymunk import Vec2d
from .cutils import LanePosition


class Road:
    def __init__(self,nLanes,width,points,hasWalkway = True):

        self.nLanes = nLanes
        self.width = width
        self.points = points
        self.hasWalkway = hasWalkway

        self.direction = self.points[1]-self.points[0]
        self.length = self.direction.length
        self.direction = self.direction / self.length
        self.normal = copy.deepcopy(self.direction)
        self.normal.rotate(math.pi/2)

        self.Lanes = [[self.points[0] + i*width*self.normal,self.points[1] + i*width*self.normal] for i in range(-nLanes,nLanes+1)]
        self.Walkways = [[self.points[0] + (nLanes+1)*width*self.normal,self.points[1] + (nLanes+1)*width*self.normal],
        [self.points[0] - (nLanes+1)*width*self.normal,self.points[1] - (nLanes+1)*width*self.normal] ]if hasWalkway else []

    def isPointOnRoad(self,point,angle):
        point -= self.points[0]
        dist = self.direction.cross(point)
        if abs(dist) > self.nLanes*self.width:
            return LanePosition.OffRoad
        else:
            pos = LanePosition.OffRoad

            dirDist = self.direction.dot(point)
            if dirDist > 0 and dirDist < self.length:
                relAngle = math.cos(self.direction.angle - angle)*dist
                pos = LanePosition.InRightLane if relAngle < 0 else LanePosition.InOpposingLane

            return pos