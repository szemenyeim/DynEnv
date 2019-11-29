from .cutils import CollisionType, friction_car, friction_car_crashed
from .cutils import LanePosition
import math
from pymunk import Vec2d, Poly, Body, moment_for_poly

class Car(object):

    # Basic parameters
    masses = [1200,1800,3500,5000]
    widths = [5,6,7,8]
    lengths = [10,15,20,25]
    powers = [3,4,3,4]
    angleDiff = math.pi/180

    def __init__(self,center,angle,type,team,goal):

        # Params based on vehicle type
        self.width = self.widths[type]
        self.height = self.lengths[type]
        mass = self.masses[type]
        self.points = [Vec2d(self.height,self. width), Vec2d(-self.height, self.width),
                       -Vec2d(self.height, self.width), Vec2d(self.height, -self.width)]
        inertia = moment_for_poly(mass,self.points)

        # Basic params
        self.type = type
        self.team = team
        self.goal = goal

        # Flags
        self.finished = False
        self.crashed = False

        # Direction
        self.direction = Vec2d(1,0)
        self.direction.rotate(angle)

        # Position
        self.position = LanePosition.OffRoad

        # For reward
        self.prevPos = center

        # Create body
        body = Body(mass, inertia, Body.DYNAMIC)
        body.position = center
        body.angle = angle
        body.velocity_func = friction_car
        self.shape = Poly(body, self.points)
        self.shape.color = (0, 255, 0)
        self.shape.elasticity = 0.05
        self.shape.collision_type = CollisionType.Car

    # Move forward or break
    def accelerate(self,dir, continuousActions):
        if not self.finished:

            power = dir

            moveDir = self.shape.body.velocity.dot(self.direction)

            # Handle continuous actions
            if continuousActions:
                # breaking is more powerful
                if dir*moveDir < 0:
                    power *= 2
                # Reverse is less powerful
                elif dir < 0:
                    power *= 0.75
            # Handle categorical actions
            else:
                # Reverse is less powerful
                if dir < 0:
                    power = dir*0.75

                # When breaking accelerate in the opposite direction (breaking is more powerful)
                if dir == 0:
                    power = 0 if not moveDir else (-2 if moveDir > 0 else 2)
                # Accelerating in the opposite direction has no effect
                elif dir < 0 and moveDir > 0:
                    return
                elif dir > 0 and moveDir < 0:
                    return

            # Get velocity
            velocity = Vec2d(self.powers[self.type]*power,0)
            velocity.rotate(self.shape.body.angle)

            # Add to previous
            self.shape.body.velocity = self.shape.body.velocity + velocity

            # Prevent the car from moving backwards when breaking
            if dir == 0 and self.shape.body.velocity.dot(self.direction)*moveDir < 0:
                self.shape.body.velocity = Vec2d(0,0)

    # Turn
    def turn(self,dir):
        if not self.finished:

            # Get direction and add rotate body
            rot = dir*self.angleDiff
            self.shape.body.angle += rot
            self.direction.rotate(rot)

            # Rotate velocity
            vel = self.shape.body.velocity
            vel.rotate(rot)
            self.shape.body.velocity = vel

    # Crash
    def crash(self):
        self.shape.color = (255, 10, 10)
        self.finished = True
        self.crashed = True

        # Increase friction (should stop fast)
        self.shape.body.velocity_func = friction_car_crashed

    # Getter for position
    def getPos(self):
        return self.shape.body.position

    # Getter for corners
    def getPoints(self):
        center = self.getPos()
        return [p+center for p in self.points]

    # Getter for angle
    def getAngle(self):
        return self.shape.body.angle