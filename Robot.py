import pymunk
from utils import *
# Needs id, team, fallen, penalized, orientation, head orientation
# Penalize, fall, getup
# Step forward, sideways, turn, kick
# Callback for collision for other robots: fall, pushing, ball - kicker, goalpost - falling
# Move callback for leaving the field
# Vision

class Robot(object):
    def __init__(self,x,y,team):
        mass = 10000
        a = (-10,0)
        b = (10,0)
        radius = 7.5
        inertia = pymunk.moment_for_segment(mass,a,b,radius)
        body = pymunk.Body(mass, inertia, pymunk.Body.DYNAMIC)
        body.position = x, y

        body.velocity_func = friction_robot
        self.shape = pymunk.Segment(body,a,b,radius)#pymunk.Poly(body,vertices)
        self.shape.color = (255, int(255*(1-team)), int(255*team))
        self.shape.elasticity = 0.5
        self.shape.friction = 1.5
        self.team = team
        self.velocity = 100
        self.penalized = False
        self.penalTime = 0
        self.moving = False
        self.moveTime = 0
        self.kicking = False
        self.leftFoot = True
        self.initPos = None

    def step(self, dir):
        if not self.moving and not self.penalized:
            self.moveTime = 50
            self.moving = True
            velocity = None
            if dir == 0:
                velocity = pymunk.Vec2d(0,self.velocity)
            elif dir == 1:
                velocity = pymunk.Vec2d(0,-self.velocity)
            elif dir == 2:
                velocity = pymunk.Vec2d(2*self.velocity,0)
            elif dir == 3:
                velocity = pymunk.Vec2d(-self.velocity,0)
            if velocity is not None:
                self.shape.body.velocity = velocity
                self.leftFoot = not self.leftFoot


    def turn(self, dir):
        if not self.moving and not self.penalized:
            self.moving = True
            self.moveTime = 50
            if dir == 0:
                self.shape.body.angular_velocity += 5
            elif dir == 1:
                self.shape.body.angular_velocity -= 5

    def kick(self, foot):
        if not self.moving and not self.penalized:
            self.initPos = self.shape.body.position
            self.moving = True
            self.kicking = True
            self.moveTime = 50
            self.shape.body.velocity = pymunk.Vec2d(self.velocity*7,0)

    def tick(self,time):
        if self.moving:
            self.moveTime -= time
            if self.kicking and self.moveTime + time > 25 and self.moveTime <= 25:
                print("back")
                self.shape.body.velocity = pymunk.Vec2d(self.velocity*7,0)
            if self.moveTime <= 0:
                self.moveTime = 0
                self.moving = False
                if self.kicking:
                    self.kicking = False
                    self.shape.body.position = self.initPos
                self.shape.body.velocity = pymunk.Vec2d(0,0)
                self.shape.body.angular_velocity = 0.0
        if self.penalized:
            self.penalTime -= time
            if self.penalTime <= 0:
                self.penalTime = 0
                self.penalized = False

    def isLeavingField(self):
        pos = self.shape.body.position

        outMin = 5
        outMaxX = 895
        outMaxY = 595

        if pos.y < outMin or pos.x < outMin or pos.y > outMaxY or pos.x > outMaxX:
            self.penalized = True
            self.penalTime = 5000
            self.moving = False
            x = 160 if self.team else 740
            y = 60 if pos.y < 300 else 540
            self.shape.body.position = pymunk.Vec2d(x,y)
            self.shape.body.angle = 90 if y < 300 else 270
            self.shape.body.velocity = pymunk.Vec2d(0.0,0.0)
            self.shape.body.angular_velocity = 0.0

