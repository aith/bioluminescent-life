import taichi as ti
import numpy as np



s = 10  # cell size
numrow = 10  # num cells per row
numcol = 20 # cells per col
shape = (numrow, numcol)
npa = np.random.randint(low=0, high= 2, size=shape)
a = ti.field(dtype=ti.f32, shape=shape)
a.from_numpy(npa)


# what
@ti.kernel
def paint():
    pass


canw = s * numrow
canl = s * numcol
imgw = canw
imgl = canl
res = (600, 600)

gui = ti.GUI('title', res)
while gui.running:
    a_resized = ti.imresize(a, *res).astype(np.uint8) * 1.0
    gui.set_image(a_resized)
    gui.show()