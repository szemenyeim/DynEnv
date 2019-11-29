from .cutils import CollisionType, friction_ball
from pymunk import Body, Circle, moment_for_circle, Vec2d

class Ball(object):
    def __init__(self,x,y,radius):

        # setup shape
        mass = 10
        inertia = moment_for_circle(mass, 0, radius*2, (0, 0))
        body = Body(mass, inertia)
        body.position = x, y
        body.velocity_func = friction_ball
        self.shape = Circle(body, radius*2, (0, 0))
        self.shape.color = (255, 0, 0)
        self.shape.elasticity = 0.98
        self.shape.friction = 2.5
        self.shape.collision_type = CollisionType.Ball

        # Initial and previous positions
        self.initPos = x,y
        self.prevPos = Vec2d(x,y)

        # List of robots who last touched the ball
        self.lastKicked = []

    def getPos(self):
        return self.shape.body.position
