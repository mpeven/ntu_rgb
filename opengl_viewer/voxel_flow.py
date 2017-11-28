import numpy as np
from tqdm import tqdm
from opengl_viewer.shapes import *



class Voxel_Flow_3D:
    def __init__(self, voxel_flow):
        self.voxel_flow = voxel_flow
        self.num_frames = voxel_flow.shape[0]
        self.min_displacement = 0.00001
        self.vbo_data = self.create_vbo_data()

    def get_vertices(self, frame):
        return self.vbo_data[frame][0]

    def get_colors(self, frame):
        return self.vbo_data[frame][1]

    def get_indices(self, frame):
        return self.vbo_data[frame][2]



    def create_vbo_data(self):
        frames = []
        for frame in tqdm(range(self.num_frames), "Creating VBO data - 3D voxel flow"):
            # Create arrows at optical flow vectors with length above minimum
            arrows = self.create_arrows(frame)
            num_arrow_verts = len(arrows)//3
            num_arrows = len(arrows)//arrow_verts.size
            arrow_colors = np.tile(np.array([0, 1, 0]), num_arrow_verts)
            arrow_indices = np.repeat(np.arange(num_arrows), len(arrow_idxs))*arrow_verts.shape[0] + np.tile(arrow_idxs, num_arrows)

            # Create voxels
            voxels = self.create_voxels(frame)
            num_cube_verts = len(voxels)//3
            num_voxels = len(voxels)//cube_verts.size
            colors = np.tile(np.array([1, 0, 0]), num_cube_verts)
            indices = np.repeat(np.arange(num_voxels), len(cube_idxs))*cube_verts.shape[0] + np.tile(cube_idxs, num_voxels)
            indices += max(arrow_indices, default=-1) + 1 # Start indexing after arrows

            # Combine arrows and voxels
            frames.append((np.concatenate([arrows, voxels]),
                           np.concatenate([arrow_colors, colors]),
                           np.concatenate([arrow_indices, indices])))
        return frames



    def create_voxels(self, frame_idx):
        non_zeros = np.nonzero(self.voxel_flow[frame_idx, :, :, :, 0])
        vertices = np.tile(cube_verts.copy(), (non_zeros[0].shape[0], 1))
        x = vertices[:,0] + np.repeat(non_zeros[0], 8)/100.0 - 0.5
        y = vertices[:,1] + np.repeat(non_zeros[1], 8)/-100.0 + 0.5
        z = vertices[:,2] + np.repeat(non_zeros[2], 8)/100.0 - 0.5
        vertices = np.stack([x,y,z]).T.flatten()
        return vertices



    def create_arrows(self, frame_idx):
        # Create array of optical flow vectors
        non_zeros = np.nonzero(self.voxel_flow[frame_idx, :, :, :, 0])
        num_arrows = non_zeros[0].shape[0]
        op_flow = np.zeros([num_arrows, 6])
        for idx in range(num_arrows):
            x = non_zeros[0][idx]/100.0 - 0.5
            y = non_zeros[1][idx]/-100.0 + 0.5
            z = non_zeros[2][idx]/100.0 - 0.5
            dx = self.voxel_flow[frame_idx, non_zeros[0][idx], non_zeros[1][idx], non_zeros[2][idx], 1]
            dy = -1 * self.voxel_flow[frame_idx, non_zeros[0][idx], non_zeros[1][idx], non_zeros[2][idx], 2]
            dz = self.voxel_flow[frame_idx, non_zeros[0][idx], non_zeros[1][idx], non_zeros[2][idx], 3]
            op_flow[idx] = np.array([x,y,z,dx,dy,dz])

        # Remove vectors with norm = 0
        xyz_length = np.linalg.norm(op_flow[:, 3:], axis=1)
        mask = (xyz_length > self.min_displacement)
        xyz_length = xyz_length[mask]
        op_flow = op_flow[mask]
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

        # Get starting vertices
        vertices = np.tile(arrow_verts.copy(), (num_arrows, 1))

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
