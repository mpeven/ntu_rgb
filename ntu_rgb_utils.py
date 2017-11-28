import pickle
import numpy as np
from opengl_viewer.opengl_viewer import OpenGlViewer
from ntu_rgb import NTU


dataset = NTU()

voxel_flow = dataset.get_voxel_flow(0)
viewer = OpenGlViewer(voxel_flow)

viewer.view()
