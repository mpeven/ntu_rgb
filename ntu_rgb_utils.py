import pickle
import numpy as np
from opengl_viewer.opengl_viewer import OpenGlViewer
from ntu_rgb import NTU


dataset = NTU()

all_voxels = []
for vid in range(15):
    all_voxels.append(dataset.get_voxel_flow(vid))
voxel_flow = np.concatenate(all_voxels)
print(voxel_flow.shape)
viewer = OpenGlViewer(voxel_flow, record=True)
viewer.view()
