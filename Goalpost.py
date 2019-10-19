from utils import *

class Goalpost(object):
    def __init__(self,x,y, radius):
        body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        body.position = x, y
        self.shape = pymunk.Circle(body, radius*2, (0, 0))
        self.shape.color = (0, 0, 0)
        self.shape.elasticity = 0.95
        self.shape.collision_type = collision_types["goalpost"]
