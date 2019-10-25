from pymunk import Vec2d
import math, random, copy
from enum import IntEnum


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
        noiseVec1 = Vec2d(5*(random.random()-0.5),5*(random.random()-0.5))
        noiseVec2 = Vec2d(5*(random.random()-0.5),5*(random.random()-0.5))

        # Add random noise if simple noise and random FN
        if noiseType == 1:
            if random.random() < rand:
                obj[0] = SightingType.NoSighting
            obj[1] += noiseVec1
            obj[2] += noiseVec2

        # Else add bigger noise to distant lines (also bigger probability of FN)
        elif noiseType == 2:
            multiplier = 1
            if obj[0] == SightingType.Distant:
                multiplier = 4
            elif obj[0] == SightingType.Partial:
                multiplier = 2
            if random.random() < rand*multiplier:
                obj[0] = SightingType.NoSighting
            obj[1] += noiseVec1*multiplier/2
            obj[2] += noiseVec2*multiplier/2

    return obj

# Add random noise to other sightings
def addNoise(obj,noiseType,interaction, rand, maxDist, misClass = False):

    if interaction == InteractionType.Occlude:
        obj[0] = SightingType.NoSighting
        return obj

    if noiseType and obj[0]:

        # Random position noise
        noiseVec = Vec2d(5*(random.random()-0.5),5*(random.random()-0.5))

        # Add random noise to position and size and FN
        if noiseType == 1:
            if random.random() < rand:
                obj[0] = SightingType.NoSighting
            obj[1] += noiseVec
            obj[2] *= (1-(random.random()-0.5)*0.2)

        # Realistic noise
        elif noiseType == 2:

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
            obj[2] *= 1+(random.random() * 0.05 * diff)

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