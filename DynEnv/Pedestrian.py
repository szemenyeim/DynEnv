from .cutils import CollisionType, friction_pedestrian_dead
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
        self.shape.color = (0, 255, 255)
        self.shape.collision_type = CollisionType.Pedestrian
        self.shape.elasticity = 0.05

        # Walk parameter
        self.lenRange = road.length
        self.widthRange = (road.nLanes+1)*road.width*2
        self.side = side

        # Bool flags
        self.dead = False

        # Move parameters
        self.moving = 0
        self.direction = road.direction
        self.normal = road.normal
        self.speed = random.randint(3,6)

        # Flags for crossing
        self.crossing = False
        self.beginCrossing = False

    # Die function
    def die(self):
        self.moving = 0
        self.shape.body.velocity = Vec2d(0,0)
        self.shape.color = (255, 0, 0)
        self.dead = True

        # Increase friction (should stop fast)
        self.shape.body.velocity_func = friction_pedestrian_dead

    # Getter function for position
    def getPos(self):
        return self.shape.body.position