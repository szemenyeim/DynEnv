from utils import *

class Ball(object):
    def __init__(self,x,y):
        mass = 10
        radius = 7.5
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        body.velocity_func = friction_ball
        self.shape = pymunk.Circle(body, radius, (0, 0))
        self.shape.color = (0, 0, 255)
        self.shape.elasticity = 0.95
        self.shape.friction = 1.5
        self.shape.collision_type = collision_types["ball"]
        self.lastKicked = 0

    def isOutOfField(self):
        finished = False
        reward = 0
        pos = self.shape.body.position

        outMin = 60 - self.shape.radius
        outMaxX = 840 + self.shape.radius
        outMaxY = 540 + self.shape.radius

        if pos.y < outMin or pos.x < outMin or pos.y > outMaxY or pos.x > outMaxX:
            x = 450
            y = 300
            if pos.y < outMin or pos.y > outMaxY:
                x = pos.x - 50 if self.lastKicked else pos.x + 50
                if pos.y < outMin:
                    y = outMin + self.shape.radius
                else:
                    y = outMaxY - self.shape.radius
            else:
                if pos.y < 400 and pos.y > 200:
                    finished = True
                    if pos.x < outMin:
                        reward = 1
                    else:
                        reward = -1
                else:
                    if pos.x < outMin:
                        x = outMin + 100
                    else:
                        x = outMaxX - 100
            self.shape.body.position = pymunk.Vec2d(x,y)
            self.shape.body.velocity = pymunk.Vec2d(0.0,0.0)
            self.shape.body.angular_velocity = 0.0
        return finished,reward
