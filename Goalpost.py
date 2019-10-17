import pymunk
from pymunk import *
# move callback to detect goals and out of fields

class Goalpost(object):
    def __init__(self,x,y):
        radius = 10
        body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        body.position = x, y
        self.shape = pymunk.Circle(body, radius, (0, 0))
        self.shape.color = (0, 0, 255)
        self.shape.elasticity = 0.95
        #print("ball")
