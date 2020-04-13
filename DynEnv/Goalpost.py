from .cutils import CollisionType
from pymunk import Body, Circle

class Goalpost(object):
    radius = 5
    def __init__(self,x,y, side, dir):
        body = Body(0, 0, Body.STATIC)
        body.position = x, y
        self.side = side
        self.dir = dir
        self.shape = Circle(body, self.radius*2, (0, 0))
        self.shape.color = (0, 0, 255)
        self.shape.elasticity = 0.95
        self.shape.collision_type = CollisionType.Goalpost
