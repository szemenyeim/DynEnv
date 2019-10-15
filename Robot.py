import pymunk

# Needs id, team, fallen, penalized, orientation, head orientation
# Penalize, fall, getup
# Step forward, sideways, turn, kick
# Callback for collision for other robots: fall, pushing, ball - kicker, goalpost - falling
# Move callback for leaving the field
# Vision

class Robot(object):
    def __init__(self):
        print("robot")