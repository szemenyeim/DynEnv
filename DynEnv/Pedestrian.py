from .utils import CollisionType, friction_pedestrian_dead
from pymunk import Vec2d, Circle, Body, moment_for_circle
import random


class Pedestrian(object):

    def __init__(self, center, road, side):

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
        self.side = side

        self.dead = False

        # Move direction
        self.moving = 0
        self.crossing = False
        self.beginCrossing = False
        self.direction = road.direction
        self.normal = road.normal
        self.speed = random.randint(4,10)

    def die(self):
        self.moving = 0
        self.shape.body.velocity = Vec2d(0,0)
        self.shape.color = (255, 0, 0)
        self.dead = True
        self.shape.body.velocity_func = friction_pedestrian_dead

    def getPos(self):
        return self.shape.body.position