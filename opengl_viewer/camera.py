
import numpy as np

xyz0     = np.array([-.02, .15, -2.])
forward0 = np.array([0., -0.05, 1.])
up0      = np.array([0., 1., 0.])

class Camera:
    def __init__(self):
        self.xyz     = xyz0.copy()
        self.forward = forward0.copy()
        self.up      = up0.copy()
        self.right   = np.cross(self.forward, self.up)


    def get_viewing_matrix(self):
        reference_point = self.xyz + self.forward
        all_vals = *self.xyz, *reference_point, *self.up
        return all_vals


    def reset(self):
        self.xyz     = xyz0.copy()
        self.forward = forward0.copy()
        self.up      = up0.copy()
        self.right   = np.cross(self.forward, self.up)


    def rotate_camera_right(self, angle):
        x, y, z = self.forward
        self.forward = self.forward * np.cos(angle) - self.right * np.sin(angle)
        self.forward /= np.linalg.norm(self.forward)
        self.right   = np.cross(self.forward, self.up)


    def rotate_camera_up(self, angle):
        x, y, z = self.forward
        self.forward = self.forward * np.cos(angle) + self.up * np.sin(angle)
        self.forward /= np.linalg.norm(self.forward)
        self.up = np.cross(self.forward, self.right) * -1


    def move_forward(self, step_size):
        self.xyz += step_size * self.forward


    def move_right(self, step_size):
        self.xyz += step_size * self.right
