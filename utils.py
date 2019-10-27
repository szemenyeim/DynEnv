from pymunk import Body, Vec2d
from enum import IntEnum

# Type of noise to be added
class NoiseType(IntEnum):
    Random = 0
    Realistic = 1

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

# Robot friction callback
def friction_robot(body, gravity, damping, dt):
    apply_friction(body,gravity,damping,dt,2e-3,1e-2)

# Ball friction callback
def friction_ball(body, gravity, damping, dt):
    apply_friction(body,gravity,damping,dt,4e-2,2e-3,5e-2)

def apply_friction(body, gravity, damping, dt, friction, rotFriction, spin = 0.0):

    # Update velocity
    Body.update_velocity(body, gravity, damping, dt)

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
    body.velocity = Vec2d(x,y)
    body.angular_velocity = theta