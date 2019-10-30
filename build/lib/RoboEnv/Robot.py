from .utils import CollisionType, friction_robot
from pymunk import Body, Segment, moment_for_segment, Vec2d
from pymunk.constraint import PivotJoint, RotaryLimitJoint
import math

class Robot(object):

    # Properties
    length = 10
    radius = 7.5
    totalRadius = length+radius
    fieldOfView = math.pi/4
    velocity = 50
    ang_velocity = 20
    mass = 4000

    def __init__(self,pos,team,id):

        # Foot positions
        a = (-self.length,self.length)
        b = (self.length,self.length)
        c = (-self.length,-self.length)
        d = (self.length,-self.length)

        # Setup left foot
        inertia = moment_for_segment(self.mass,a,b,self.radius)
        body = Body(self.mass, inertia, Body.DYNAMIC)
        body.position = pos
        body.angle = 0 if team > 0 else math.pi
        body.velocity_func = friction_robot
        self.leftFoot = Segment(body,a,b,self.radius)
        self.leftFoot.color = (255, int(127*(1-team)), int(127*(1+team)))
        self.leftFoot.elasticity = 0.3
        self.leftFoot.friction = 2.5
        self.leftFoot.collision_type = CollisionType.Robot

        # Setup right foot
        inertia = moment_for_segment(self.mass,c,d,self.radius)
        body = Body(self.mass, inertia, Body.DYNAMIC)
        body.position = pos
        body.angle = 0 if team > 0 else math.pi
        body.velocity_func = friction_robot
        self.rightFoot = Segment(body,c,d,self.radius)
        self.rightFoot.color = (255, int(127*(1-team)), int(127*(1+team)))
        self.rightFoot.elasticity = 0.3
        self.rightFoot.friction = 2.5
        self.rightFoot.collision_type = CollisionType.Robot

        # setup joint
        self.joint = PivotJoint(self.leftFoot.body,self.rightFoot.body,(pos[0],pos[1]))
        self.joint.error_bias = 0.1
        self.rotJoint = RotaryLimitJoint(self.leftFoot.body,self.rightFoot.body,0,0)

        # Basic properties
        self.team = team
        self.id = id
        self.headAngle = 0

        # Penalty and pushing parametes
        self.penalized = False
        self.penalTime = 0
        self.touching = False
        self.touchCntr = 0
        self.mightPush = False
        self.fallen = False
        self.fallCntr = 0

        # Movement parameters
        self.moveTime = 0
        self.headMoving = 0
        self.headMaxAngle = 2*math.pi/3

        # Kick parameters
        self.kicking = False
        self.initPos = None
        self.foot = None

    def getPos(self):
        return (self.leftFoot.body.position + self.rightFoot.body.position)/2.0

    # Move in certain direction (relative to the robot
    def step(self, dir):
        if not self.kicking and not self.penalized and not self.fallen:
            self.moveTime = 500
            velocity = None
            if dir == 0:
                velocity = Vec2d(0,2.5*self.velocity)
            elif dir == 1:
                velocity = Vec2d(0,-2.5*self.velocity)
            elif dir == 2:
                velocity = Vec2d(2.5*self.velocity,0)
            elif dir == 3:
                velocity = Vec2d(-2.5*self.velocity,0)
            if velocity is not None:
                shape = self.leftFoot
                angle = shape.body.angle
                velocity.rotate(angle)
                shape.body.velocity = velocity

    # Turn
    def turn(self, dir):
        if not self.kicking and not self.penalized and not self.fallen:
            self.moveTime = 500
            self.leftFoot.body.angular_velocity += self.ang_velocity if dir else -self.ang_velocity

    # Kick
    def kick(self, foot):
        if not self.kicking and not self.penalized and not self.fallen:
            self.foot = foot
            self.initPos = self.rightFoot.body.position if foot else self.leftFoot.body.position
            self.kicking = True
            self.moveTime = 1000

    # Turn the head
    def turnHead(self,dir):
        self.headMoving = dir*math.pi/720
        self.moveTime = 500

