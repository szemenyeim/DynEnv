from .utils import CollisionType
from pymunk import Body, Circle

class Goalpost(object):
    def __init__(self,x,y, radius):
        body = Body(0, 0, Body.STATIC)
        body.position = x, y
        self.shape = Circle(body, radius*2, (0, 0))
        self.shape.color = (0, 0, 0)
        self.shape.elasticity = 0.95
        self.shape.collision_type = CollisionType.Goalpost
