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

        self.followDist = 80

        self.direction = self.points[1]-self.points[0]
        self.length = self.direction.length
        self.direction = self.direction / self.length
        self.normal = copy.deepcopy(self.direction)
        self.normal.rotate(math.pi/2)

        self.Lanes = [[self.points[0] + i*width*self.normal,self.points[1] + i*width*self.normal] for i in range(-nLanes,nLanes+1)]
        self.Walkways = [[self.points[0] + (nLanes+1)*width*self.normal,self.points[1] + (nLanes+1)*width*self.normal],
        [self.points[0] - (nLanes+1)*width*self.normal,self.points[1] - (nLanes+1)*width*self.normal] ]if hasWalkway else []

    def isPointOnRoad(self,point,angle):
        pt = point - self.points[0]
        dist = self.direction.cross(pt)
        if abs(dist) >= self.nLanes*self.width+5:
            return LanePosition.OffRoad
        else:
            pos = LanePosition.OffRoad

            dirDist = self.direction.dot(pt)
            if dirDist >= -10 and dirDist <= self.length+10:
                relAngle = math.cos(self.direction.angle - angle)*dist
                pos = LanePosition.InRightLane if relAngle < 0 else LanePosition.InOpposingLane

            return pos

    def getSpot(self,lane,spot):
        end = 1 if lane >= self.nLanes else 0
        pos = self.points[end]
        spotDir = (-self.direction if end else self.direction) * self.followDist
        laneDir = (self.normal if end else -self.normal) * self.width

        lane = (lane-self.nLanes if end else lane)+0.5

        return pos + lane*laneDir + spot*spotDir,spotDir.angle

    def getWalkSpot(self,side,length,width):
        center = self.Walkways[side][0] + length*(self.Walkways[side][1]-self.Walkways[side][0])
        center += width*self.width*self.normal*(1 if side else -1)
        return center