import numpy as np
from tqdm import tqdm
from opengl_viewer.shapes import *




class Optical_flow_3D:
    def __init__(self, optical_flow):
        self.op_flow = optical_flow
        self.min_op_flow_length = 0.00001
        self.num_frames = len(optical_flow)
        self.vbo_data = self.create_vbo_data()

    def get_vertices(self, frame):
        return self.vbo_data[frame][0]

    def get_colors(self, frame):
        return self.vbo_data[frame][1]

    def get_indices(self, frame):
        return self.vbo_data[frame][2]

    def create_vbo_data(self):
        frames = []
        for frame in tqdm(range(self.num_frames), "Creating VBO data"):
            # Turn y and dy upsidedown
            self.op_flow[frame][:,1] *= -1
            self.op_flow[frame][:,4] *= -1

            # Create arrows at optical flow vectors with length above minimum
            arrows = self.create_arrows(self.op_flow[frame].copy())
            num_arrow_verts = len(arrows)//3
            num_arrows = len(arrows)//arrow_verts.size
            arrow_colors = np.tile(np.array([1, 0, 0]), num_arrow_verts)
            arrow_indices = np.repeat(np.arange(num_arrows), len(arrow_idxs))*arrow_verts.shape[0] + np.tile(arrow_idxs, num_arrows)

            # Create pyramids at optical flow vectors with length below minimum
            pyramids = self.create_pyramids(self.op_flow[frame][:].copy())
            num_pyramid_verts = len(pyramids)//3
            num_pyramids = len(pyramids)//pyramid_verts.size
            colors = np.tile(np.array([.5, .5, 1]), num_pyramid_verts)
            indices = np.repeat(np.arange(num_pyramids), len(pyramid_idxs))*4 + np.tile(pyramid_idxs, num_pyramids)
            indices += max(arrow_indices, default=-1) + 1

            # Combine arrows and pyramids
            frames.append((np.concatenate([arrows, pyramids]),
                           np.concatenate([arrow_colors, colors]),
                           np.concatenate([arrow_indices, indices])))
        return frames

    def create_pyramids(self, op_flow):
        '''
        If the 3D optical flow isn't large enough, just render a pyramid showing
        position of the person in the depth value
        '''
        xyz_length = np.linalg.norm(op_flow[:, 3:], axis=1)
        mask = (xyz_length <= self.min_op_flow_length)
        xyz_length = xyz_length[mask]
        op_flow = op_flow[mask]
        vertices = np.tile(pyramid_verts.copy(), (op_flow.shape[0], 1))
        x = vertices[:,0] + np.repeat(op_flow[:,0], 4)
        y = vertices[:,1] + np.repeat(op_flow[:,1], 4)
        z = vertices[:,2] + np.repeat(op_flow[:,2], 4)
        vertices = np.stack([x,y,z]).T.flatten()
        return vertices


    def create_arrows(self, op_flow):
        # Remove vectors with norm = 0
        xyz_length = np.linalg.norm(op_flow[:, 3:], axis=1)
        mask = (xyz_length > self.min_op_flow_length)
        xyz_length = xyz_length[mask]
        op_flow = op_flow[mask]

        # Amount of arrows in this frame
        num_arrows = op_flow.shape[0]

        # Get rotation angles
        xy_length = np.linalg.norm(op_flow[:, 3:5], axis=1)
        mask = (xy_length == 0)
        z_angles = np.zeros([num_arrows])
        z_angles[mask] = np.radians(90)
        z_angles[~mask] = -np.arccos(op_flow[~mask, 4] / xy_length[~mask])
        x_angles = np.arccos(np.linalg.norm(op_flow[:, 3:5], axis=1) / xyz_length)
        z_angles[op_flow[:,3] < 0] *= -1
        x_angles[op_flow[:,5] < 0] *= -1

        # Perform gaussian smoothing over xyz_length
        # xyz_length = gaussian_filter(xyz_length, 4.0)

        # Get starting vertices
        vertices = np.tile(arrow_verts.copy(), (num_arrows, 1))

        # Scale arrows with size of the vector length
        # vertices *= np.repeat(xyz_length/np.average(xyz_length), len(vertices))[:,np.newaxis]

        # Add vector length to arrow tip
        vertices[8::9,1] += xyz_length*1.5

        # Rotate all points in direction of 3D vector
        x_angles = np.repeat(x_angles,9)
        z_angles = np.repeat(z_angles,9)
        cos_x = np.cos(x_angles)
        sin_x = np.sin(x_angles)
        cos_z = np.cos(z_angles)
        sin_z = np.cos(z_angles)
        y1 = vertices[:, 1]*cos_x - vertices[:, 2]*sin_x
        z1 = vertices[:, 1]*sin_x + vertices[:, 2]*cos_x
        x2 = vertices[:, 0]*cos_z - y1*sin_z
        y2 = vertices[:, 0]*sin_z + y1*cos_z
        x3 = x2 + np.repeat(op_flow[:,0], 9)
        y3 = y2 + np.repeat(op_flow[:,1], 9)
        z3 = z1 + np.repeat(op_flow[:,2], 9)
        vertices = np.stack([x3,y3,z3]).T.flatten()
        return vertices
