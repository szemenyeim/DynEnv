from .cutils import CollisionType
from pymunk import Vec2d, Poly, Body


class Obstacle(object):

    def __init__(self, center, width, height):

        # Create body
        body = Body(0, 0, Body.STATIC)
        body.position = center
        points = [Vec2d(width,height),Vec2d(-width,height),-Vec2d(width,height),Vec2d(width,-height)]
        self.width = width
        self.height = height
        self.points = [p+center for p in points]
        self.shape = Poly(body,points)
        self.shape.color = (200, 200, 200)
        self.shape.elasticity = 0.05
        self.shape.collision_type = CollisionType.Obstacle

    # Getter function for position
    def getPos(self):
        return self.shape.body.position