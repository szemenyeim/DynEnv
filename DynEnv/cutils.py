from pymunk import Vec2d
import math, random, copy
from enum import IntEnum
import numpy as np
from .utils import NoiseType

class LanePosition(IntEnum):
    AtGoal = 0
    InRightLane = 1
    InOpposingLane = 2
    OffRoad = 3

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

# Camera parameters
focal = 543.6
A = np.array([
    [focal, 0, 319.5],
    [0, -focal, 239.5],
    [0,0,1]
])

# 2*pi
twoPi = math.pi*2

# Camera orientation and rotation
bottomAng = (0.6929+0.25)
bottomRot = np.array([
        [1, 0, 0],
        [0, np.cos(bottomAng), -np.sin(bottomAng)],
        [0, np.sin(bottomAng), np.cos(bottomAng)]
    ])

bottomTr = np.concatenate((np.concatenate((bottomRot, np.array([[0,], [53.774,], [5.071,]])), axis=1),np.array([[0,0,0,1,],])),axis=0)
bottomTr = np.matmul(A,np.linalg.inv(bottomTr)[:3])

# Top camera
topAng = (0.0209+0.25)
topRot = np.array([
        [1, 0, 0],
        [0, np.cos(topAng), -np.sin(topAng)],
        [0, np.sin(topAng), np.cos(topAng)]
    ])

topTr = np.concatenate((np.concatenate((topRot, np.array([[0,], [58.364,], [5.871,]])), axis=1),np.array([[0,0,0,1,],])),axis=0)
topTr = np.matmul(A,np.linalg.inv(topTr)[:3])

angleNoise = math.pi/180

def projectPoints(points,compRadius = True):

    # Project points
    topProj = np.matmul(topTr, points)
    bottomProj = np.matmul(bottomTr, points)

    # Return from homogeneous
    topProj = topProj[0:2] / topProj[2]
    bottomProj = bottomProj[0:2] / bottomProj[2]

    # Compute radius between first and second vectors
    tRad = 0
    bRad = 0
    if compRadius:
        tRad = np.sqrt(np.sum(np.square(topProj[:, 0] - topProj[:, 1])))
        bRad = np.sqrt(np.sum(np.square(bottomProj[:, 0] - bottomProj[:, 1])))

    return topProj,math.ceil(tRad),bottomProj,math.ceil(bRad)

# Get x coordinates on a conic defined by params for y values in a range between [0-yRange)
def getConicPoints(yRange,center,params):

    # y Coordinates of the image (transform to 0 center conic)
    yCoord = np.arange(0,yRange)-center[1]

    # Precompute 4*a and 1/2a
    a = params[0]
    a4 = 4*a
    overa = 1.0/(2*a)

    # Create b and c coefficients for every y
    b = yCoord*params[2] + params[3]
    c = yCoord*(yCoord*params[1] + params[4])+params[5]

    # Compute determinants
    sqr = b*b-a4*c
    ind = sqr >= 0

    # Solve equations for nonnegative determinants
    sqrt = np.sqrt(sqr[ind])
    x1 = np.round((-b[ind] + sqrt)*overa+center[0])
    x2 = np.round((-b[ind] - sqrt)*overa+center[0])

    # Create image coorinates
    yCoord = np.round(yCoord + center[1])
    c1 = np.vstack( (x1, yCoord[ind]) ).astype('int32').transpose()
    c2 = np.vstack( (x2, yCoord[ind]) ).astype('int32').transpose()

    return c1,c2

# Estimate conic parameters from five points
def estimateConic(points):
    # Tranpose points
    points = points.transpose()

    # Setup parameter matrix: Every row is [x^2, y^2, xy, x, y, 1]
    X = np.array([
        [points[0,0]*points[0,0],points[0,1]*points[0,1],points[0,0]*points[0,1],points[0,0],points[0,1],1],
        [points[1,0]*points[1,0],points[1,1]*points[1,1],points[1,0]*points[1,1],points[1,0],points[1,1],1],
        [points[2,0]*points[2,0],points[2,1]*points[2,1],points[2,0]*points[2,1],points[2,0],points[2,1],1],
        [points[3,0]*points[3,0],points[3,1]*points[3,1],points[3,0]*points[3,1],points[3,0],points[3,1],1],
        [points[4,0]*points[4,0],points[4,1]*points[4,1],points[4,0]*points[4,1],points[4,0],points[4,1],1],
        [points[5,0]*points[5,0],points[5,1]*points[5,1],points[5,0]*points[5,1],points[5,0],points[5,1],1],
        [points[6,0]*points[6,0],points[6,1]*points[6,1],points[6,0]*points[6,1],points[6,0],points[6,1],1],
        [points[7,0]*points[7,0],points[7,1]*points[7,1],points[7,0]*points[7,1],points[7,0],points[7,1],1]
    ])

    # Solve equation
    u,s,vt = np.linalg.svd(X)
    a = -vt[-1]
    return a

# Class color scheme for visualization
classColors = [
    (0,0,255),
    (0,255,0),
    (255,0,0),
    (255,255,255),
]

# Visualization function
def colorize(img):
    cImg = np.zeros((img.shape[1],img.shape[2],3)).astype('uint8')
    for i in range(0,len(classColors)):
        ind = len(classColors)-1-i
        cImg[img[ind]==1] = classColors[ind]
    return cImg

def normalize(pt,normFactor, mean = None):
    if mean is None:
        mean = 0
    return ((pt*normFactor)-mean)*2

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
            multiplier1 = 0.25 + 3.75*obj[1].get_length_sqrd()/maxDist
            multiplier2 = 0.25 + 3.75*obj[2].get_length_sqrd()/maxDist
            multiplier = (multiplier1+multiplier2)*0.5

            if random.random() < rand*multiplier:
                obj[0] = SightingType.NoSighting

            obj[1] += noiseVec1*multiplier1/2
            obj[2] += noiseVec2*multiplier2/2


# Add noise to lane
def addNoiseLane(obj,noiseType, magn, rand, maxDist):

    if obj[0]:

        # Create random distance and angle noise
        distNoise = 1 - (random.random() - 0.5) * magn * 0.1
        angleNoise = (random.random() - 0.5) * magn * 0.1

        # Add random noise if simple noise and random FN
        if noiseType == NoiseType.Random:
            if random.random() < rand:
                obj[0] = SightingType.NoSighting
            obj[1] *= distNoise
            obj[2] += angleNoise

        # Else add bigger noise to distant lines (also bigger probability of FN)
        elif noiseType == NoiseType.Realistic:

            # Get distances of points and average dist
            multiplier1 = 0.25 + 3.75 * obj[1]*obj[1] / maxDist

            if random.random() < rand * multiplier1:
                obj[0] = SightingType.NoSighting

            obj[1] *= distNoise * multiplier1 / 2
            obj[2] += angleNoise * multiplier1 / 2


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
            range = 0.25 + 3.75*obj[1].get_length_sqrd()/maxDist
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

# Function to change sighting type of occluded objects
def filterOcclude(obj,interaction):
    if interaction == InteractionType.Occlude:
        obj[0] = SightingType.NoSighting
    return obj

# Add noise to polynom
def addNoiseRect(obj,noiseType,interaction, magn, rand, maxDist, misClass = False):

    if obj[0]:

        # Random position noise
        noiseVec = Vec2d((random.random()-0.5),(random.random()-0.5))*magn

        # Add random noise to position and size and FN
        if noiseType == NoiseType.Random:

            # Change to FN
            if random.random() < rand:
                obj[0] = SightingType.NoSighting
            else:
                newPos = obj[1] + noiseVec
                # Compute rotation
                angleDiff = (random.random()-0.5)*magn*angleNoise
                obj[2] += angleDiff
                # Center corner points and rotate
                if obj[3] is not None:
                    obj[3] = [pt-obj[1] for pt in obj[3]]
                    [pt.rotate(angleDiff) for pt in obj[3]]
                    obj[3] = [pt+obj[1] for pt in obj[3]]
                # Compute new center and add it to corners
                obj[1]  = newPos

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

            # Random misclassification if the flag is set
            if random.random() < rand*multiplier:
                obj[0] = SightingType.NoSighting
                return
            if misClass and random.random() < rand*multiplier/2:
                obj[0] = SightingType.Misclassified

            # Apply noise
            angleDiff = (random.random()-0.5)*magn*angleNoise
            obj[2] += angleDiff

            # Center corners and rotate
            if obj[3] is not None:
                obj[3] = [pt-obj[1] for pt in obj[3]]
                [pt.rotate(angleDiff) for pt in obj[3]]
                obj[3] = [pt+newPos for pt in obj[3]]
            # Compute new center and add it to corners
            obj[1] = newPos

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
        if abs(dist) < radius and obj1.get_length_sqrd() < obj2.get_length_sqrd():
            type = InteractionType.Occlude

    return type

# Is object seen in radius
def isSeenInRadius(point,corners,angle,obsPt,obsAngle,maxDist,distantDist):

    # Center point and get distance
    trPt = point-obsPt
    dist = trPt.get_length_sqrd()

    # Decide normal and distant sighting
    if dist <= maxDist:
        seen = SightingType.Distant
        if dist <= distantDist:
            seen = SightingType.Normal

        if corners is not None:
            # Center corners and rotate with object angle
            corners = [corner - point for corner in corners]
            if angle != 0:
                [corner.rotate(angle) for corner in corners]

            # Add back transformaed center and rotate with observer angle
            corners = [corner + trPt for corner in corners]
            [corner.rotate(-obsAngle) for corner in corners]

        # Rotate center point and get new object angle
        trPt.rotate(-obsAngle)
        trAngle = angle - obsAngle

        return [seen,trPt,trAngle,corners]


    return [SightingType.NoSighting,]


def angle(corner):

    return math.atan2(corner.y,corner.x)


def getLineInRadius(points,obsPt,obsAngle,maxDist):

    dist = (obsPt-points[0]).cross(points[1]-points[0])/(obsPt-points[0]).length
    if dist*dist > maxDist:
        return [SightingType.NoSighting,]
    ang = math.cos(angle(points[1]-points[0])-obsAngle)
    return [SightingType.Normal, dist, ang]

    # Center points
    '''trPts = [point-obsPt for point in points]

    # Chortcuts
    x1 = trPts[0].x
    y1 = trPts[0].y
    dx = trPts[1].x-x1
    dy = trPts[1].y-y1

    # Compute coefficients
    a = dx*dx+dy*dy
    b = 2*(x1*dx + y1*dy)
    c = x1*x1+y1*y1-maxDist

    # Determinant
    det = b*b-4*a*c

    # Solve equation
    if det >= 0:
        den = 0.5/a
        sqrDet = math.sqrt(det)
        t1 = (-b-sqrDet)*den
        t2 = (-b+sqrDet)*den
        dP = trPts[1]-trPts[0]

        # Get points
        trPts = [trPts[0] + t1*dP,trPts[0] + t2*dP]
        # Rotate them
        [pt.rotate(-obsAngle) for pt in trPts]

        return [SightingType.Normal,trPts[0],trPts[1]]

    return[SightingType.NoSighting,]'''

def getViewBlockAngle(centerAngle,corners):

    # Get relative angles and distances
    angles = np.array([angle(corner)-centerAngle for corner in corners])
    distances = np.array([corner.get_length_sqrd() for corner in corners])

    # Transform angles into the +/- pi interval
    angles[angles > math.pi] -= twoPi
    angles[angles < -math.pi] += twoPi

    # Get minmax angles and closest point
    minIdx = np.argmin(angles)
    maxIdx = np.argmax(angles)
    cIdx = np.argmin(distances)

    return angles,minIdx,maxIdx,cIdx

def doesInteractPoly(elem1,elem2,radius,canOcclude=True):

    # Default value
    ret = InteractionType.NoInter

    # Return if theay are not seen
    if elem1[0] == SightingType.NoSighting or elem2[0] == SightingType.NoSighting:
        return ret

    # Get variables
    point1 = elem1[1]
    point2 = elem2[1]
    corners = elem2[3]

    # Check proximity
    if radius > 0 and (point2-point1).get_length_sqrd() < radius:
        ret = InteractionType.Nearby

    if canOcclude:

        angle2 = angle(point2)

        # Get blocked interval
        angles,minIdx,maxIdx,closestIdx = getViewBlockAngle(angle2,corners)

        # Get min and max angles
        minAngle = angles[minIdx]
        maxAngle = angles[maxIdx]

        # Get extreme points and closest one
        p1 = corners[minIdx]
        p2 = corners[maxIdx]
        pm = corners[closestIdx]

        # Angle difference between centers
        pAngle = angle(point1) - angle2

        # Normalize angle to +/- pi interval
        if pAngle > math.pi:
            pAngle -= twoPi
        elif pAngle < -math.pi:
            pAngle += twoPi

        # If object falls into angle range
        if pAngle > minAngle and pAngle < maxAngle:
            # If one of the extreme points is the closest
            if closestIdx == minIdx or closestIdx == maxIdx:
                # Check if objects is on the far side of the line between extreme points
                if ((p2-p1).cross(point1-p1) < 0):
                    ret = InteractionType.Occlude
            # Otherwise it needs to be on the far side of two lines
            elif ((p2-pm).cross(point1-pm) < 0) and ((pm-p1).cross(point1-p1) < 0):
                ret = InteractionType.Occlude

    return ret

def isSeenInArea(point,dir1,dir2,maxDist,angle,radius=0,allowPartial=True):

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
            if point.get_length_sqrd() < maxDist:
                seen = SightingType.Normal
            else:
                seen = SightingType.Distant
        elif allowPartial:
            seen = SightingType.Partial
        # Compute circle line segment intersection
        else:
            # Setup b and c of the quadratic equation (a=1 in this case)
            b1 = -2 * (dir1.x * point.x - dir1.y * point.y)
            b2 = -2 * (dir2.x * point.x - dir2.y * point.y)
            c = point.dot(point) - radius * radius

            # Compute determinants
            sqr1 = b1 * b1 - 4 * c
            sqr2 = b2 * b2 - 4 * c

            # If the intersection is on the positive part of the line, we have a partial sigting
            found = False
            if sqr1 >= 0:
                sqrt = math.sqrt(sqr1)
                if (-b1 + sqrt) > 0 or (-b1 - sqrt) > 0:
                    seen = SightingType.Partial
                    found = True
            if not found and sqr2 >= 0:
                sqrt = math.sqrt(sqr2)
                if (-b2 + sqrt) > 0 or (-b2 - sqrt) > 0:
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
            if pt1.get_length_sqrd() > maxDist or pt2.get_length_sqrd() > maxDist:
                seen = SightingType.Distant

            # Transfer the points to the robot coordinate system
            if pt1 and pt2:
                pt1.rotate(-angle)
                pt2.rotate(-angle)

            # Check points behind the robot
            if pt1.x < 0 or pt2.x < 0:
                seen = SightingType.NoSighting

    return [seen,pt1,pt2]