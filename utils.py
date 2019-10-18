import pymunk

collision_types = {
    "ball": 1,
    "goalpost": 2,
    "robot": 3,
}

def friction_robot(body, gravity, damping, dt):
    apply_friction(body,gravity,damping,dt,1e-3,1e-2)

def friction_ball(body, gravity, damping, dt):
    apply_friction(body,gravity,damping,dt,5e-2,2e-3,5e-2)

def apply_friction(body, gravity, damping, dt, friction, rotFriction, spin = 0.0):

    pymunk.Body.update_velocity(body, gravity, damping, dt)
    m = body.mass
    factor = friction*m
    rotFactor = rotFriction*m

    x = body.velocity.x
    y = body.velocity.y
    length = abs(x)+abs(y) + 1e-5
    theta = body.angular_velocity

    a = [x*factor/length, y*factor/length]
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