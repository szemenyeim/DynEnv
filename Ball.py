from utils import *

class Ball(object):
    def __init__(self,x,y,radius):
        mass = 10
        inertia = pymunk.moment_for_circle(mass, 0, radius*2, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        body.velocity_func = friction_ball
        self.shape = pymunk.Circle(body, radius*2, (0, 0))
        self.shape.color = (0, 0, 255)
        self.shape.elasticity = 0.98
        self.shape.friction = 2.5
        self.shape.collision_type = collision_types["ball"]
        self.lastKicked = 0

    def isOutOfField(self,space):
        finished = False
        reward = 0
        pos = self.shape.body.position

        outMin = space.sideLength - self.shape.radius
        outMaxX = space.W - space.sideLength + self.shape.radius
        outMaxY = space.H -space.sideLength + self.shape.radius

        if pos.y < outMin or pos.x < outMin or pos.y > outMaxY or pos.x > outMaxX:
            x = space.W/2
            y = space.H/2
            if pos.y < outMin or pos.y > outMaxY:
                x = pos.x + 50 if self.lastKicked else pos.x - 50
                if pos.y < outMin:
                    y = outMin + self.shape.radius
                else:
                    y = outMaxY - self.shape.radius
            else:
                if pos.y < space.H/2 + space.goalWidth and pos.y > space.H/2 - space.goalWidth:
                    finished = True
                    if pos.x < outMin:
                        reward = -1
                    else:
                        reward = 1
                else:
                    if pos.x < outMin:
                        if self.lastKicked:
                            x = space.sideLength + space.penaltyLength
                        else:
                            x = space.sideLength
                            y = space.sideLength if pos.y < space.H/2 else space.H-space.sideLength
                    else:
                        if not self.lastKicked:
                            x = space.W - (space.sideLength + space.penaltyLength)
                        else:
                            x = space.W - space.sideLength
                            y = space.sideLength if pos.y < space.H/2 else space.H-space.sideLength
            self.shape.body.position = pymunk.Vec2d(x,y)
            self.shape.body.velocity = pymunk.Vec2d(0.0,0.0)
            self.shape.body.angular_velocity = 0.0
        return finished,reward
