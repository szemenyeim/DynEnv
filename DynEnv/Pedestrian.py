from .utils import CollisionType
from pymunk import Vec2d, Circle, Body, moment_for_circle


class Pedestrian(object):

    def __init__(self, center):

        # setup shape
        mass = 90
        radius = 2.5
        inertia = moment_for_circle(mass, 0, radius * 2, (0, 0))
        body = Body(mass, inertia)
        body.position = center
        self.shape = Circle(body, radius * 2, (0, 0))
        self.shape.color = (0, 0, 255)
        self.shape.collision_type = CollisionType.Pedestrian

        # Move direction
        self.moveTime = 15000
        self.moving = 0

    def move(self,time):
        if self.moving > 0:
            self.moving = max(0,self.moving-time)
        else:
            self.moving = self.moveTime