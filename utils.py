import pymunk
import copy
import math
import random
from enum import IntEnum

class GameType(IntEnum):
    GetBall = 0
    Full = 1

class NoiseType(IntEnum):
    Noiseless = 0
    Random = 1
    Realistic = 2

class ObservationType(IntEnum):
    Full = 0
    Partial = 1
    Image = 2

class CollisionType(IntEnum):
    Ball = 0
    Goalpost = 1
    Robot = 2

class SightingType(IntEnum):
    NoSighting = 0
    Partial = 1
    Distant = 2
    Normal = 3
    Misclassified = 4

class InteractionType(IntEnum):
    NoInter = 0
    Nearby = 1
    Occlude = 2

def addNoiseLine(obj,noiseType, rand):
    if noiseType and obj[0]:
        noiseVec1 = pymunk.Vec2d(10*(random.random()-0.5),10*(random.random()-0.5))
        noiseVec2 = pymunk.Vec2d(10*(random.random()-0.5),10*(random.random()-0.5))
        if noiseType == 1:
            return (obj[0] if random.random() > rand else SightingType.NoSighting,obj[1]+noiseVec1,obj[2]+noiseVec2)
        elif noiseType == 2:
            multiplier = 1 if obj[0] == SightingType.Normal else 2
            return (obj[0] if random.random() > rand else SightingType.NoSighting,obj[1]+noiseVec1*multiplier,obj[2]+noiseVec2*multiplier)
    return obj


def addNoise(obj,noiseType,interaction, rand, misClass = False):
    if noiseType and obj[0]:
        noiseVec = pymunk.Vec2d(10*(random.random()-0.5),10*(random.random()-0.5))
        if noiseType == 1:
            newObj = (obj[0] if random.random() > rand else SightingType.NoSighting,obj[1]+noiseVec,obj[2]+(random.random()-0.5)*2)
            if len(obj) == 4:
                newObj = newObj + (obj[3],)
            return newObj
        elif noiseType == 2:
            sightingType = obj[0]
            multiplier = 1 if sightingType == SightingType.Normal and interaction == InteractionType.NoInter else 2
            newPos = obj[1]+noiseVec*multiplier
            diff = +1 if newPos.length-obj[1].length > 0 else -1
            if misClass and random.random() < rand:
                sightingType = SightingType.Misclassified
            newObj =  (sightingType if random.random() > rand*multiplier else SightingType.NoSighting,obj[1]+noiseVec*multiplier,obj[2]+random.random()*2*diff)
            if len(obj) == 4:
                newObj = newObj + (obj[3],)
            return newObj
    return obj

def doesInteract(obj1,obj2,radius,canOcclude=True):
    if obj2 is None or obj1 is None:
        return InteractionType.NoInter

    type = InteractionType.NoInter
    if (obj1-obj2).length < radius:
        type = InteractionType.Nearby

    if canOcclude:
        dist = obj1.cross(obj2)/obj1.length
        if abs(dist) < radius and obj1.length < obj2.length:
            type = InteractionType.Occlude

    return type

def isSeenInArea(point,dir1,dir2,maxDist,radius=0):
    seen = SightingType.NoSighting
    rotPt = None
    if point.length < maxDist:
        dist1 = dir1.cross(point)
        dist2 = dir2.cross(point)
        if dist1 < radius and dist2 > -radius:
            angle = (dir1.angle + dir2.angle)*0.5
            if dist1 < -radius and dist2 > radius:
                if point.length < maxDist*0.75:
                    seen = SightingType.Normal
                else:
                    seen = SightingType.Distant
            else:
                seen = SightingType.Partial
            rotPt = copy.copy(point)
            rotPt.rotate(angle)
    return seen,rotPt,radius

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

def isLineInArea(p1,p2,dir1,dir2,maxDist,maxDistSqr):
    seen = SightingType.NoSighting
    pt1 = None
    pt2 = None
    dist11 = dir1.cross(p1)
    dist12 = dir1.cross(p2)
    if not (dist11 > 0 and dist12 > 0):
        dist21 = dir2.cross(p1)
        dist22 = dir2.cross(p2)
        if not (dist21 < 0 and dist22 < 0):
            angle = (dir1.angle + dir2.angle)*0.5
            seen = SightingType.Normal
            if dist11 <= 0 and dist21 >= 0:
                pt1 = copy.copy(p1)
            else:
                inter1 = p1.cross(dir1)/dir1.cross(p2-p1)
                inter2 = p1.cross(dir2)/dir2.cross(p2-p1)
                inter = inter1 if abs(inter1) > abs(inter2) else inter2
                pt1 = p1+inter*(p2-p1)
                seen = SightingType.Partial
            if dist12 <= 0 and dist22 >= 0:
                pt2 = copy.copy(p2)
            else:
                inter1 = p2.cross(dir1)/dir1.cross(p1-p2)
                inter2 = p2.cross(dir2)/dir2.cross(p1-p2)
                inter = inter1 if abs(inter1) > abs(inter2) else inter2
                pt2 = p2+inter*(p1-p2)
                seen = SightingType.Partial
            if pt1.length > maxDist:
                if pt2.length > maxDist:
                    seen = SightingType.NoSighting
                    pt1 = None
                    pt2 = None
                else:
                    pt1 = getLine(pt1,pt2,maxDistSqr)
                    seen = SightingType.Distant
            elif pt2.length > maxDist:
                pt2 = getLine(pt2, pt1, maxDistSqr)
                seen = SightingType.Distant
            if pt1 and pt2:
                pt1.rotate(angle)
                pt2.rotate(angle)
    return seen,pt1,pt2

def friction_robot(body, gravity, damping, dt):
    apply_friction(body,gravity,damping,dt,2e-3,1e-2)

def friction_ball(body, gravity, damping, dt):
    apply_friction(body,gravity,damping,dt,4e-2,2e-3,5e-2)

def apply_friction(body, gravity, damping, dt, friction, rotFriction, spin = 0.0):

    pymunk.Body.update_velocity(body, gravity, damping, dt)
    m = body.mass
    factor = friction*m
    rotFactor = rotFriction*m

    x = body.velocity.x
    y = body.velocity.y
    length = 1.0/(abs(x)+abs(y) + 1e-5)
    theta = body.angular_velocity

    a = [x*factor*length, y*factor*length]
    a[0] += a[1]*spin*theta
    a[1] -= a[0]*spin*theta

    if abs(x) < factor:
        x = 0
    else:
        x -= a[0]
    if abs(y) < factor:
        y = 0
    else:
        y -= a[1]

    if abs(theta) < rotFactor:
        theta = 0
    else:
        theta -= rotFactor if theta > 0 else -rotFactor

    body.velocity = pymunk.Vec2d(x,y)
    body.angular_velocity = theta