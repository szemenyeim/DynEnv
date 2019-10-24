import pymunk
import copy
import math
import random
from enum import IntEnum

# Type of noise to be added
class NoiseType(IntEnum):
    Noiseless = 0
    Random = 1
    Realistic = 2

# Observation types
class ObservationType(IntEnum):
    Full = 0
    Partial = 1
    Image = 2

# Enum for object collision handling
class CollisionType(IntEnum):
    Ball = 0
    Goalpost = 1
    Robot = 2

# Type of observation
class SightingType(IntEnum):
    NoSighting = 0
    Partial = 1
    Distant = 2
    Normal = 3
    Misclassified = 4

# Types of object-ocject interaction
class InteractionType(IntEnum):
    NoInter = 0
    Nearby = 1
    Occlude = 2

# Add noise to a line sighting
def addNoiseLine(obj,noiseType, rand):

    if noiseType and obj[0]:

        # Create random position noise
        noiseVec1 = pymunk.Vec2d(10*(random.random()-0.5),10*(random.random()-0.5))
        noiseVec2 = pymunk.Vec2d(10*(random.random()-0.5),10*(random.random()-0.5))

        # Add random noise if simple noise and random FN
        if noiseType == 1:
            return (obj[0] if random.random() > rand else SightingType.NoSighting,obj[1]+noiseVec1,obj[2]+noiseVec2)

        # Else add bigger noise to distant lines (also bigger probability of FN)
        elif noiseType == 2:
            multiplier = 1 if obj[0] == SightingType.Normal else 2
            return (obj[0] if random.random() > rand*multiplier else SightingType.NoSighting,obj[1]+noiseVec1*multiplier,obj[2]+noiseVec2*multiplier)

    return obj

# Add random noise to other sightings
def addNoise(obj,noiseType,interaction, rand, misClass = False):
    if noiseType and obj[0]:

        # Random position noise
        noiseVec = pymunk.Vec2d(10*(random.random()-0.5),10*(random.random()-0.5))

        # Add random noise to position and size and FN
        if noiseType == 1:
            newObj = (obj[0] if random.random() > rand else SightingType.NoSighting,obj[1]+noiseVec,obj[2]+(random.random()-0.5)*2)

            # Robots have more properties, let's not leave them out
            if len(obj) == 4:
                newObj = newObj + obj[3:]

            return newObj

        # Realistic noise
        elif noiseType == 2:

            # Add larger noise to distant objetcs
            sightingType = obj[0]
            multiplier = 1 if sightingType == SightingType.Normal and interaction == InteractionType.NoInter else 2
            newPos = obj[1]+noiseVec*multiplier

            # positive if object moved farther
            diff = +1 if newPos.length-obj[1].length > 0 else -1

            # Random misclassification if the flag is set
            if misClass and random.random() < rand:
                sightingType = SightingType.Misclassified

            # The sign of the noise on the object size is determined by whether it moved closer due to position noise
            newObj =  (sightingType if random.random() > rand*multiplier else SightingType.NoSighting,obj[1]+noiseVec*multiplier,obj[2]+random.random()*2*diff)

            # Robots have more properties, let's not leave them out
            if len(obj) >= 4:
                newObj = newObj + obj[3:]

            return newObj

    return obj

# Is there interaction between the two sightings
def doesInteract(obj1,obj2,radius,canOcclude=True):
    if obj2 is None or obj1 is None:
        return InteractionType.NoInter

    # Proximity check
    type = InteractionType.NoInter
    if (obj1-obj2).length < radius:
        type = InteractionType.Nearby

    # Check for occlusions
    if canOcclude:

        # Distance between the line going towards ob1 (LoS) and the position of obj2
        dist = obj1.cross(obj2)/obj1.length

        # If obj2 falls in the LoS and is closer, there is occlusion
        if abs(dist) < radius and obj1.length < obj2.length:
            type = InteractionType.Occlude

    return type

def isSeenInArea(point,dir1,dir2,maxDist,radius=0):

    # Get object sighting
    seen = SightingType.NoSighting
    rotPt = None

    # If close enough
    if point.length < maxDist:

        # Get signed distances from the field of view lines
        dist1 = dir1.cross(point)
        dist2 = dir2.cross(point)

        # If inside the field of view
        if dist1 < radius and dist2 > -radius:

            # Get robot orientation
            angle = (dir1.angle + dir2.angle)*0.5

            # Check for partial sightings
            if dist1 < -radius and dist2 > radius:

                # Check for distant sightings
                if point.length < maxDist*0.75:
                    seen = SightingType.Normal
                else:
                    seen = SightingType.Distant
            else:
                seen = SightingType.Partial

            # Rotate sighting in the robot's coordinate system
            rotPt = copy.copy(point)
            rotPt.rotate(angle)

    return seen,rotPt,radius

# pt1 is in the FoV but too far
# This functions returns the farthest point that is within viewing distance
def getLine(pt1,pt2,maxDistSqr):
    xSqr = pt2.x * pt2.x
    ySqr = pt2.y * pt2.y
    abx = pt2.x * pt1.x
    aby = pt2.y * pt1.y
    a = pt1.x*pt1.x -2*abx + xSqr + pt1.y*pt1.y - 2*aby + ySqr
    b = 2 * ((abx - xSqr) + (aby - ySqr))
    c = xSqr + ySqr - maxDistSqr
    den = 1.0 / (2 * a)
    sqr = math.sqrt(b ** 2 - 4 * a * c)
    k1 = (-b + sqr) * den
    k2 = (-b - sqr) * den
    return pt2 + max(k1, k2) * (pt1 - pt2)

# Checks if line is visible
def isLineInArea(p1,p2,dir1,dir2,maxDist,maxDistSqr):

    # Defaults
    seen = SightingType.NoSighting

    # Seen endpoints
    pt1 = None
    pt2 = None

    # Signed distance from one edge of the FoV
    dist11 = dir1.cross(p1)
    dist12 = dir1.cross(p2)

    # If both ends are outside, we can't see the line
    if not (dist11 > 0 and dist12 > 0):

        # Same for the other edge
        dist21 = dir2.cross(p1)
        dist22 = dir2.cross(p2)
        if not (dist21 < 0 and dist22 < 0):

            # Robot orientation
            angle = (dir1.angle + dir2.angle)*0.5

            # At this point, we probably see the line
            seen = SightingType.Normal

            # If p1 is inside, we see it
            if dist11 <= 0 and dist21 >= 0:
                pt1 = copy.copy(p1)
            # Othervise, compute intersection of FoV and the p1-p2 line
            else:
                # Check which line intersects closer to p1
                inter1 = p1.cross(dir1)/dir1.cross(p2-p1)
                inter2 = p1.cross(dir2)/dir2.cross(p2-p1)
                inter = inter1 if abs(inter1) > abs(inter2) else inter2

                # Compute the intesection position
                pt1 = p1+inter*(p2-p1)
                seen = SightingType.Partial

            # If p2 is inside, we see it
            if dist12 <= 0 and dist22 >= 0:
                pt2 = copy.copy(p2)
            # Othervise, compute intersection of FoV and the p2-p1 line
            else:
                # Check which line intersects closer to p2
                inter1 = p2.cross(dir1)/dir1.cross(p1-p2)
                inter2 = p2.cross(dir2)/dir2.cross(p1-p2)
                inter = inter1 if abs(inter1) > abs(inter2) else inter2

                # Compute the intesection position
                pt2 = p2+inter*(p1-p2)
                seen = SightingType.Partial

            # Are the points close enough
            if pt1.length > maxDist:
                if pt2.length > maxDist:
                    # If both are too far, we don't see the line after all
                    seen = SightingType.NoSighting
                    pt1 = None
                    pt2 = None
                # If we see pt2 but not pt1
                else:
                    # Get the farthest point on the line we can see
                    pt1 = getLine(pt1,pt2,maxDistSqr)
                    seen = SightingType.Distant
            # If we see pt1 but not pt2
            elif pt2.length > maxDist:
                # Get the farthest point on the line we can see
                pt2 = getLine(pt2, pt1, maxDistSqr)
                seen = SightingType.Distant

            # Transfer the points to the robot coordinate system
            if pt1 and pt2:
                pt1.rotate(angle)
                pt2.rotate(angle)

    return seen,pt1,pt2

# Robot friction callback
def friction_robot(body, gravity, damping, dt):
    apply_friction(body,gravity,damping,dt,2e-3,1e-2)

# Ball friction callback
def friction_ball(body, gravity, damping, dt):
    apply_friction(body,gravity,damping,dt,4e-2,2e-3,5e-2)

def apply_friction(body, gravity, damping, dt, friction, rotFriction, spin = 0.0):

    # Update velocity
    pymunk.Body.update_velocity(body, gravity, damping, dt)

    # Get friction coefficients for velocity and angular velocity
    m = body.mass
    factor = friction*m
    rotFactor = rotFriction*m

    # Get velocity params
    x = body.velocity.x
    y = body.velocity.y
    length = 1.0/(abs(x)+abs(y) + 1e-5)
    theta = body.angular_velocity

    # Spinning objects turn sideways
    a = [x*factor*length, y*factor*length]
    a[0] += a[1]*spin*theta
    a[1] -= a[0]*spin*theta

    # If too slow, simply stop
    if abs(x) < factor:
        x = 0
    else:
        x -= a[0]
    if abs(y) < factor:
        y = 0
    else:
        y -= a[1]

    # Ditto
    if abs(theta) < rotFactor:
        theta = 0
    else:
        theta -= rotFactor if theta > 0 else -rotFactor

    # Set velocity
    body.velocity = pymunk.Vec2d(x,y)
    body.angular_velocity = theta