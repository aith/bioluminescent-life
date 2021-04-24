import taichi as ti
import numpy as np
import clingo as cl
ti.init()

@ti.kernel
def init():
    pass
    # for i, j in alive:
    #     if ti.random() > 0.8:
    #         alive[i, j] = 1
    #     else:
    #         alive[i, j] = 0


img_size = 800
gui = ti.GUI('Title', (img_size, img_size)); gui.fps_limit = 15
# cell_size = gui.slider('cell size', 0.0, 255.0, 1.0); cell_size.value = 2.0

n = img_size
grid = ti.field(int, shape=(n, n))
init()
while gui.running:
    # for e in gui.get_events(gui.PRESS, gui.MOTION):
    #     if e.key == gui.ESCAPE:
    #         gui.running = False
    #     elif e.key == gui.SPACE:
    #         pass
    # if gui.is_pressed(gui.LMB, gui.RMB):
    #     # mx, my = gui.get_cursor_pos()
    #     # alive[int(mx * n), int(my * n)] = gui.is_pressed(gui.LMB)
    #     # paused = True
    #     pass
    # if not paused:
    #     run()

    gui.set_image(ti.imresize(grid, img_size).astype(np.uint8) * 255)
    gui.show()