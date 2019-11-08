from .utils import CollisionType
from pymunk import Vec2d, Poly, Body


class Obstacle(object):

    def __init__(self, center, width, height):
        body = Body(0, 0, Body.STATIC)
        body.position = center

        points = [center+Vec2d(width,height),center+Vec2d(-width,height),center-Vec2d(width,height),center+Vec2d(width,-height)]
        self.shape = Poly(body,points)
        self.shape.color = (255, 0, 0)
        self.shape.elasticity = 0.05
        self.shape.collision_type = CollisionType.Obstacle