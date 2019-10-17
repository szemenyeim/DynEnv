from Control import *

# Launch game, allow user controls

if __name__ == '__main__':
    env = Environment()
    action = 0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.key == K_w:
                    action = 1
                elif event.key == K_s:
                    action = 2
                elif event.key == K_d:
                    action = 3
                elif event.key == K_a:
                    action = 4
                elif event.key == K_q:
                    action = 5
                elif event.key == K_e:
                    action = 6
                elif event.key == K_r:
                    action = 7
            elif event.type == KEYUP:
                action = 0

        ret = env.step(action)
        if ret[1]:
            print("Goal: reward: ",ret[0])