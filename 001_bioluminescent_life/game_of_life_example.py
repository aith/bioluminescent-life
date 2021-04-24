import taichi as ti
import numpy as np
import clingo as cl
ti.init()



@ti.func
def get_count(i, j):  # get count of neighbors
    return (alive[i - 1, j] + alive[i + 1, j] + alive[i, j - 1] +
            alive[i, j + 1] + alive[i - 1, j - 1] + alive[i + 1, j - 1] +
            alive[i - 1, j + 1] + alive[i + 1, j + 1])


# See https://www.conwaylife.com/wiki/Cellular_automaton#Rules for more rules
B, S = [3], [2, 3]
#B, S = [2], [0]


@ti.func
def calc_rule(a, c):
    if a == 0:
        for t in ti.static(B):
            if c == t:
                a = 1
    elif a == 1:
        a = 0
        for t in ti.static(S):
            if c == t:
                a = 1
    return a


@ti.kernel
def run():
    for i, j in alive:
        count[i, j] = get_count(i, j)

    for i, j in alive:
        alive[i, j] = calc_rule(alive[i, j], count[i, j])


@ti.kernel
def init():
    for i, j in alive:
        if ti.random() > 0.8:
            alive[i, j] = 1
        else:
            alive[i, j] = 0


n = 256
cell_size = 4
img_size = n * cell_size
alive = ti.field(int, shape=(n, n))  # alive = 1, dead = 0
count = ti.field(int, shape=(n, n))  # count of neighbours

gui = ti.GUI('Game of Life', (img_size, img_size))
cell_size = gui.slider('cell size', 0.0, 255.0, 1.0); cell_size.value = 2.0
gui.fps_limit = 15

init()
paused = False
while gui.running:
    for e in gui.get_events(gui.PRESS, gui.MOTION):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.SPACE:
            paused = not paused
        elif e.key == 'r':
            alive.fill(0)

    if gui.is_pressed(gui.LMB, gui.RMB):
        mx, my = gui.get_cursor_pos()
        alive[int(mx * n), int(my * n)] = gui.is_pressed(gui.LMB)
        # paused = True

    if not paused:
        run()

    gui.set_image(ti.imresize(alive, img_size).astype(np.uint8) * 255)
    gui.show()