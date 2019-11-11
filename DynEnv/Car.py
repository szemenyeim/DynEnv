from .utils import CollisionType, friction_car, friction_car_crashed
from .cutils import LanePosition
import math
from pymunk import Vec2d, Poly, Body, moment_for_poly

class Car(object):

    masses = [1200,1800,3500,5000]
    widths = [5,6,7,8]
    lengths = [10,15,20,25]
    powers = [3,4,3,4]

    def __init__(self,center,angle,type,team,goal):

        width = self.widths[type]
        height = self.lengths[type]
        mass = self.masses[type]
        points = [Vec2d(height, width), Vec2d(-height, width), -Vec2d(height, width), Vec2d(height, -width)]
        inertia = moment_for_poly(mass,points)

        self.type = type
        self.team = team
        self.goal = goal

        self.finished = False
        self.crashed = False

        self.direction = Vec2d(1,0)
        self.direction.rotate(angle)

        self.position = LanePosition.OffRoad

        self.prevPos = center

        body = Body(mass, inertia, Body.DYNAMIC)
        body.position = center
        body.angle = angle
        body.velocity_func = friction_car
        self.shape = Poly(body, points)
        self.shape.color = (0, 255, 0)
        self.shape.elasticity = 0.05
        self.shape.collision_type = CollisionType.Car

        self.angleDiff = math.pi/90

    def accelerate(self,dir):
        if not self.finished:
            velocity = Vec2d(self.powers[self.type]*dir,0)
            velocity.rotate(self.shape.body.angle)
            self.shape.body.velocity = self.shape.body.velocity + velocity
            if self.shape.body.velocity.dot(self.direction) < 0:
                self.shape.body.velocity = Vec2d(0,0)

    def turn(self,dir):
        if not self.finished:
            rot = dir*self.angleDiff
            self.shape.body.angle += rot
            vel = self.shape.body.velocity
            vel.rotate(rot)
            self.shape.body.velocity = vel
            self.direction.rotate(rot)

    def crash(self):
        self.shape.color = (255, 0, 0)
        self.finished = True
        self.crashed = True
        self.shape.body.velocity_func = friction_car_crashed