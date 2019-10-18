from utils import *
import math

class Robot(object):
    def __init__(self,x,y,team):
        mass = 5000
        a = (-10,10)
        b = (10,10)
        c = (-10,-10)
        d = (10,-10)
        radius = 7.5
        inertia = pymunk.moment_for_segment(mass,a,b,radius)
        body = pymunk.Body(mass, inertia, pymunk.Body.DYNAMIC)
        body.position = x, y
        body.angle = 0 if team else math.pi
        body.velocity_func = friction_robot

        self.leftFoot = pymunk.Segment(body,a,b,radius)
        self.leftFoot.color = (255, int(255*(1-team)), int(255*team))
        self.leftFoot.elasticity = 0.5
        self.leftFoot.friction = 1.5
        self.leftFoot.collision_type = collision_types["robot"]
        inertia = pymunk.moment_for_segment(mass,c,d,radius)
        body = pymunk.Body(mass, inertia, pymunk.Body.DYNAMIC)
        body.position = x, y
        body.angle = 0 if team else math.pi
        body.velocity_func = friction_robot
        self.rightFoot = pymunk.Segment(body,c,d,radius)
        self.rightFoot.color = (255, int(255*(1-team)), int(255*team))
        self.rightFoot.elasticity = 0.5
        self.rightFoot.friction = 1.5
        self.rightFoot.collision_type = collision_types["robot"]

        self.spring = pymunk.constraint.PivotJoint(self.leftFoot.body,self.rightFoot.body,(x,y))
        self.spring.error_bias = 0.1
        self.rotSpring = pymunk.constraint.RotaryLimitJoint(self.leftFoot.body,self.rightFoot.body,0,0)
        self.spring.collide_bodies = False

        self.team = team
        self.velocity = 50
        self.penalized = False
        self.penalTime = 0
        self.moving = False
        self.moveTime = 0
        self.kicking = False
        self.initPos = None
        self.fallen = False
        self.mightPush = False
        self.fallCntr = 0
        self.touching = False

    def step(self, dir):
        if not self.moving and not self.penalized:
            self.moveTime = 500
            self.moving = True
            velocity = None
            if dir == 0:
                velocity = pymunk.Vec2d(0,2*self.velocity)
            elif dir == 1:
                velocity = pymunk.Vec2d(0,-2*self.velocity)
            elif dir == 2:
                velocity = pymunk.Vec2d(2*self.velocity,0)
            elif dir == 3:
                velocity = pymunk.Vec2d(-self.velocity,0)
            if velocity is not None:
                shape = self.leftFoot
                angle = shape.body.angle
                velocity.rotate(angle)
                shape.body.velocity = velocity


    def turn(self, dir):
        if not self.moving and not self.penalized:
            self.moving = True
            self.moveTime = 500
            if dir == 0:
                self.leftFoot.body.angular_velocity += 20
            elif dir == 1:
                self.leftFoot.body.angular_velocity -= 20

    def kick(self):
        if not self.moving and not self.penalized:
            self.initPos = self.leftFoot.body.position
            self.moving = True
            self.kicking = True
            self.moveTime = 1000

    def fall(self,space):
        print("Fall", self.team)

        pos = (self.leftFoot.body.position + self.rightFoot.body.position)/2.0
        filter = pymunk.shape_filter.ShapeFilter(categories=0b101)
        shapes = space.point_query(pos,30,filter)

        for query in shapes:
            if query.shape != self.leftFoot and query.shape != self.rightFoot:
                force = self.velocity*self.leftFoot.body.mass*query.shape.body.mass/100.0
                dp = pos - query.shape.body.position
                dp = -dp*force/dp.length
                query.shape.body.apply_force_at_world_point(dp,pos)

        self.leftFoot.color = (255, int(100*(1-self.team)), int(100*self.team))
        self.rightFoot.color = (255, int(100*(1-self.team)), int(100*self.team))
        self.fallen = True
        self.moving = True
        self.touching = True
        self.fallCntr += 1
        self.moveTime = 3000
        if self.fallCntr > 2:
            print("Fallen robot", self.team)
            self.penalize(5000)

    def penalize(self,time):
        print("Penalized")
        self.penalized = True
        self.penalTime = time
        self.moving = False
        pos = (self.leftFoot.body.position + self.rightFoot.body.position)/2.0
        x = 160 if self.team else 740
        y = 60 if pos.y < 300 else 540
        self.leftFoot.body.position = pymunk.Vec2d(x-10, y)
        self.rightFoot.body.position = pymunk.Vec2d(x+10, y)
        self.leftFoot.body.angle = math.pi / 2 if y < 300 else -math.pi / 2
        self.leftFoot.body.velocity = pymunk.Vec2d(0.0, 0.0)
        self.leftFoot.body.angular_velocity = 0.0
        self.leftFoot.color = (255, 0, 0)
        self.rightFoot.body.angle = math.pi / 2 if y < 300 else -math.pi / 2
        self.rightFoot.body.velocity = pymunk.Vec2d(0.0, 0.0)
        self.rightFoot.body.angular_velocity = 0.0
        self.rightFoot.color = (255, 0, 0)

    def tick(self,time,ballPos,space):
        if self.moving:
            self.moveTime -= time
            if self.kicking:
                if self.moveTime + time > 500 and self.moveTime <= 500:
                    velocity = pymunk.Vec2d(self.velocity * 4, 0)
                    angle = self.leftFoot.body.angle
                    velocity.rotate(angle)
                    self.leftFoot.body.velocity = velocity
                if self.moveTime + time > 450 and self.moveTime <= 450:
                    velocity = pymunk.Vec2d(self.velocity*4,0)
                    angle = self.leftFoot.body.angle
                    velocity.rotate(angle)
                    self.leftFoot.body.velocity = -velocity
                elif self.moveTime <= 400:
                    self.leftFoot.body.velocity = pymunk.Vec2d(0,0)
                    self.kicking = False
                    self.leftFoot.body.position = self.initPos
            if self.moveTime <= 0:
                self.moveTime = 0
                self.moving = False
                self.leftFoot.body.velocity = pymunk.Vec2d(0,0)
                self.leftFoot.body.angular_velocity = 0.0
                self.rightFoot.body.velocity = pymunk.Vec2d(0,0)
                self.rightFoot.body.angular_velocity = 0.0
                if self.fallen:
                    print("Getup", self.team)
                    self.leftFoot.color = (255, int(255*(1-self.team)), int(255*self.team))
                    self.rightFoot.color = (255, int(255*(1-self.team)), int(255*self.team))
                    self.fallen = False
                    self.fallCntr = 0
        if self.penalized:
            self.penalTime -= time
            if self.penalTime <= 0:
                print("Unpenalized")
                self.penalTime = 0
                self.penalized = False
                pos = self.leftFoot.body.position
                pos.y = 60 if ballPos.y > 300 else 540
                self.leftFoot.body.angle = math.pi / 2 if ballPos.y > 300 else -math.pi / 2
                self.leftFoot.body.position = pos
                self.leftFoot.color = (255, int(255*(1-self.team)), int(255*self.team))
                pos = self.rightFoot.body.position
                pos.y = 60 if ballPos.y > 300 else 540
                self.rightFoot.body.angle = math.pi / 2 if ballPos.y > 300 else -math.pi / 2
                self.rightFoot.body.position = pos
                self.rightFoot.color = (255, int(255*(1-self.team)), int(255*self.team))

    def isLeavingField(self):
        pos = (self.leftFoot.body.position + self.rightFoot.body.position)/2.0

        outMin = 5
        outMaxX = 895
        outMaxY = 595

        if pos.y < outMin or pos.x < outMin or pos.y > outMaxY or pos.x > outMaxX:
            self.penalize(5000)

