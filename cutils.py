from pymunk import Vec2d
import math, random, copy
from enum import IntEnum
from utils import *
import numpy as np

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

def rotX(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def rotY(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rotZ(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def rotFast(rot,ang):
    return np.matmul(rot,rotY(ang))

def rotMtx(xAng,yAng,zAng):
    return np.matmul(rotZ(zAng),np.matmul(rotY(yAng),rotX(xAng)))

# Camera parameters
A = np.array([
    [543.6, 0, 319.5],
    [0, 543.6, 239.5],
    [0,0,1]
])

# Camera orientation and rotation
bottomAng = (0.6929+0.15)
bottomRot = rotX(bottomAng)
bottomTr = np.concatenate((np.concatenate((bottomRot, np.array([[0,], [49.774,], [5.071,]])), axis=1),np.array([[0,0,0,1,],])),axis=0)
bottomTr = np.matmul(A,np.linalg.inv(bottomTr)[:3])

topAng = (0.0209+0.15)
topRot = rotX(topAng)
topTr = np.concatenate((np.concatenate((topRot, np.array([[0,], [54.364,], [5.871,]])), axis=1),np.array([[0,0,0,1,],])),axis=0)
topTr = np.matmul(A,np.linalg.inv(topTr)[:3])

def projectPoints(points,compRadius = True,allRadius=False):
    topProj = np.matmul(topTr, points)
    topProj = topProj[0:2] / topProj[2]
    bottomProj = np.matmul(bottomTr, points)
    bottomProj = bottomProj[0:2] / bottomProj[2]
    if compRadius:
        if allRadius:
            tRad = np.sqrt(np.sum(np.square(topProj[:, 0:1] - topProj[:, 1:]),axis=0))
            bRad = np.sqrt(np.sum(np.square(bottomProj[:, 0:1] - bottomProj[:, 1:]),axis=0))
        else:
            tRad = np.sqrt(np.sum(np.square(topProj[:, 0] - topProj[:, 1])))
            bRad = np.sqrt(np.sum(np.square(bottomProj[:, 0] - bottomProj[:, 1])))
    else:
        tRad = 0
        bRad = 0


    return topProj,tRad,bottomProj,bRad

Y = np.array([1,1,1])

def getEllipse(center,points):
    points = (points-center).transpose()/640.0
    X = np.array([
        [points[0,0]**2,points[0,1]**2,points[0,0]*points[0,1]*2],
        [points[1,0]**2,points[1,1]**2,points[1,0]*points[1,1]*2],
        [points[2,0]**2,points[2,1]**2,points[2,0]*points[2,1]*2]
    ])
    coeffs = np.linalg.solve(X,Y)
    theta = 0.5*math.atan2(2*coeffs[2],coeffs[1]-coeffs[0])
    nu = coeffs[0]+coeffs[1]
    zeta = (coeffs[1]-coeffs[0])/math.cos(2*theta)
    if zeta < nu:
        a = 0
    else:
        a = math.sqrt(2/(zeta-nu))
    b = math.sqrt(2/(nu+zeta))
    return theta,a,b

# Add noise to a line sighting
def addNoiseLine(obj,noiseType, magn, rand, maxDist):

    if obj[0]:

        # Create random position noise
        noiseVec1 = Vec2d((random.random()-0.5),(random.random()-0.5))*magn
        noiseVec2 = Vec2d((random.random()-0.5),(random.random()-0.5))*magn

        # Add random noise if simple noise and random FN
        if noiseType == NoiseType.Random:
            if random.random() < rand:
                obj[0] = SightingType.NoSighting
            obj[1] += noiseVec1
            obj[2] += noiseVec2

        # Else add bigger noise to distant lines (also bigger probability of FN)
        elif noiseType == NoiseType.Realistic:

            # Get distances of points and average dist
            multiplier1 = 0.25 + 3.75*obj[1].length/maxDist
            multiplier2 = 0.25 + 3.75*obj[2].length/maxDist
            multiplier = (multiplier1+multiplier2)*0.5

            if random.random() < rand*multiplier:
                obj[0] = SightingType.NoSighting

            obj[1] += noiseVec1*multiplier1/2
            obj[2] += noiseVec2*multiplier2/2

    return obj

# Add random noise to other sightings
def addNoise(obj,noiseType,interaction, magn, rand, maxDist, misClass = False):

    if interaction == InteractionType.Occlude:
        obj[0] = SightingType.NoSighting
        return obj

    if obj[0]:

        # Random position noise
        noiseVec = Vec2d((random.random()-0.5),(random.random()-0.5))*magn

        # Add random noise to position and size and FN
        if noiseType == NoiseType.Random:
            if random.random() < rand:
                obj[0] = SightingType.NoSighting
            obj[1] += noiseVec
            obj[2] *= (1-(random.random()-0.5)*0.2)

        # Realistic noise
        elif noiseType == NoiseType.Realistic:

            # Add larger noise to distant objetcs
            sightingType = obj[0]
            range = 0.25 + 3.75*obj[1].length/maxDist
            multiplier = range
            if interaction == InteractionType.Nearby:
                multiplier= range*2
            if sightingType == SightingType.Distant:
                multiplier = range*3
            elif sightingType == sightingType.Partial:
                multiplier = range*4
            newPos = obj[1]+noiseVec*multiplier/4

            # positive if object moved farther
            diff = (newPos.length-obj[1].length)

            # Random misclassification if the flag is set
            if random.random() < rand*multiplier:
                sightingType = SightingType.NoSighting
            if misClass and random.random() < rand*multiplier/2:
                sightingType = SightingType.Misclassified

            # The sign of the noise on the object size is determined by whether it moved closer due to position noise
            obj[0] = sightingType
            obj[1] = newPos
            obj[2] *= 1+(random.random() * 0.1 * diff)

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

def isSeenInArea(point,dir1,dir2,maxDist,angle,radius=0):

    # Get object sighting
    seen = SightingType.NoSighting
    rotPt = None

    # Get signed distances from the field of view lines
    dist1 = dir1.cross(point)
    dist2 = dir2.cross(point)

    # If inside the field of view
    if dist1 < radius and dist2 > -radius:

        # Check for partial sightings
        if dist1 < -radius and dist2 > radius:

            # Check for distant sightings
            if point.length < maxDist:
                seen = SightingType.Normal
            else:
                seen = SightingType.Distant
        else:
            seen = SightingType.Partial

        # Rotate sighting in the robot's coordinate system
        rotPt = copy.copy(point)
        rotPt.rotate(-angle)

    return [seen,rotPt,radius]

# Checks if line is visible
def isLineInArea(p1,p2,dir1,dir2,maxDist,angle):

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
                if inter1 < 1 and inter2 < 1:
                    inter = max(inter1,inter2)
                else:
                    inter = min(inter1,inter2)

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
                if inter1 < 1 and inter2 < 1:
                    inter = max(inter1,inter2)
                else:
                    inter = min(inter1,inter2)

                # Compute the intesection position
                pt2 = p2+inter*(p1-p2)
                seen = SightingType.Partial

            # Are the points close enough
            if pt1.length > maxDist or pt2.length > maxDist:
                seen = SightingType.Distant

            # Transfer the points to the robot coordinate system
            if pt1 and pt2:
                pt1.rotate(-angle)
                pt2.rotate(-angle)

            # Check points behind the robot
            if pt1.x < 0 or pt2.x < 0:
                seen = SightingType.NoSighting

    return [seen,pt1,pt2]