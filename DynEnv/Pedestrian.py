from .utils import CollisionType
from pymunk import Vec2d, Circle, Body, moment_for_circle
import random


class Pedestrian(object):

    def __init__(self, center, road):

        # setup shape
        mass = 90
        radius = 2.5
        inertia = moment_for_circle(mass, 0, radius * 2, (0, 0))
        body = Body(mass, inertia)
        body.position = center
        self.shape = Circle(body, radius * 2, (0, 0))
        self.shape.color = (0, 0, 255)
        self.shape.collision_type = CollisionType.Pedestrian
        self.shape.elasticity = 0.05

        self.lenRange = road.length
        self.widthRange = (road.nLanes+1)*road.width*2

        # Move direction
        self.moving = 0
        self.direction = road.direction
        self.normal = road.normal
        self.speed = random.randint(5,15)

    def move(self,time):
        if self.moving > 0:
            self.moving = max(0,self.moving-time)
        else:
            self.moving = random.randint(5000,30000)
            self.shape.body.velocity = self.speed*self.direction*random.randint(-1,1)