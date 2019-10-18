from Control import *

# Launch game, allow user controls

if __name__ == '__main__':
    env = Environment()
    action1 = 0
    action2 = 0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.key == K_w:
                    action1 = 1
                elif event.key == K_s:
                    action1 = 2
                elif event.key == K_d:
                    action1 = 3
                elif event.key == K_a:
                    action1 = 4
                elif event.key == K_q:
                    action1 = 5
                elif event.key == K_e:
                    action1 = 6
                elif event.key == K_r:
                    action1 = 7
                elif event.key == K_UP:
                    action2 = 1
                elif event.key == K_DOWN:
                    action2 = 2
                elif event.key == K_LEFT:
                    action2 = 3
                elif event.key == K_RIGHT:
                    action2 = 4
                elif event.key == K_DELETE:
                    action2 = 5
                elif event.key == K_PAGEDOWN:
                    action2 = 6
                elif event.key == K_END:
                    action2 = 7
            elif event.type == KEYUP:
                action1 = 0
                action2 = 0

        action = [action1,action2]
        ret = env.step(action)
        if ret[1]:
            print("Goal: reward: ",ret[0])