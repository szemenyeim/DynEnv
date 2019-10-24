from utils import *

class Ball(object):
    def __init__(self,x,y,radius):
        mass = 10
        inertia = pymunk.moment_for_circle(mass, 0, radius*2, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        body.velocity_func = friction_ball
        self.initPos = x,y
        self.prevPos = pymunk.Vec2d(x,y)
        self.shape = pymunk.Circle(body, radius*2, (0, 0))
        self.shape.color = (0, 0, 255)
        self.shape.elasticity = 0.98
        self.shape.friction = 2.5
        self.shape.collision_type = CollisionType.Ball
        self.lastKicked = []
