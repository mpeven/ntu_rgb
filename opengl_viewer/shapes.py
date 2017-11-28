import numpy as np


#######
# Arrow
#
s = 0.001
arrow_idxs = np.array([
    0, 1, 2, 2, 3, 0,   0, 4, 5, 5, 1, 0,   1, 5, 6, 6, 2, 1,
    2, 6, 7, 7, 3, 2,   3, 7, 4, 4, 0, 3,   4, 7, 6, 6, 5, 4,
    3, 2, 8,   2, 6, 8,   6, 7, 8,   7, 3, 8,
])
arrow_verts = np.array([
    [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
    [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s], [ 0,  s,  0],
])



#########
# Pyramid
#
s = 0.0025
pyramid_idxs = np.array([0, 1, 2, 0, 1, 3, 1, 2, 3, 2, 0, 3,])
pyramid_verts = np.array([[0, 0, s],[s, 0, -s],[-s, 0, -s],[0, -s, 0],])



############
# Cube shape
#
s = 0.0025
cube_idxs = np.array([
    0, 1, 2, 2, 3, 0,   0, 4, 5, 5, 1, 0,   1, 5, 6, 6, 2, 1,
    2, 6, 7, 7, 3, 2,   3, 7, 4, 4, 0, 3,   4, 7, 6, 6, 5, 4,
])
cube_verts = np.array([
    [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
    [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
])
