'''
NTU RGB+D Action Recognition Dataset helper

website: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
github: https://github.com/shahroudy/NTURGB-D



Functions:
- load
    loads metadata into a pickle file (only needs to be called once)
- get_rgb_vid_images
- get_ir_vid_images
- get_depth_images
    gets a numpy array of all the values for each frame in a video in the dataset
- get_point_clouds
    gets all point
- get_2D_optical_flow
    gets 2D optical flow maps
-
'''
import os, glob
import numpy as np
import numba as nb
import pandas as pd
import cv2
import av
import scipy, scipy.optimize, scipy.ndimage
import line_profiler
import datetime as dt
import re
from tqdm import tqdm
from prompter import yesno
import pickle



##################################################
# Dataset paths
rgb_vid_dir      = '/hdd/Datasets/NTU/nturgb+d_rgb'
ir_vid_dir       = '/hdd/Datasets/NTU/nturgb+d_ir'
depth_dir        = '/hdd/Datasets/NTU/nturgb+d_depth'
masked_depth_dir = '/hdd/Datasets/NTU/nturgb+d_depth_masked'
skeleton_dir     = '/hdd/Datasets/NTU/nturgb+d_skeletons'



##################################################
# Regex for pulling metadata out of video name
compiled_regex = re.compile('.*S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3}).*')



##################################################
# Subject ids used in original paper for training
train_IDs = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]



##################################################
# Some pre-defined Kinect v2 parameters

#  RGB Intrinsic Parameters
fx_rgb = 1.0820918205955327e+03
fy_rgb = 1.0823759977994873e+03
cx_rgb = 9.6692449993169430e+02
cy_rgb = 5.3566594698237566e+02

# Depth Intrinsic Parameters
fx_d = 365.481
fy_d = 365.481
cx_d = 257.346
cy_d = 210.347

# Distortion parameters
k1 = 0.0905474
k2 = -0.26819
k3 = 0.0950862
p1 = 0.
p2 = 0.

# Matrix versions
rgb_mat = np.array([[fx_rgb,     0., cx_rgb],
                    [    0., fy_rgb, cy_rgb],
                    [    0.,     0.,     1.]])
d_mat   = np.array([[fx_d,   0., cx_d],
                    [  0., fy_d, cy_d],
                    [  0.,   0.,   1.]])
dist_array = np.array([k1, k2, k3, p1, p2])





class NTU:
    def __init__(self):
        self.num_vids              = len(self.get_files(rgb_vid_dir))
        self.rgb_vids              = self.get_files(rgb_vid_dir)
        self.ir_vids               = self.get_files(ir_vid_dir)
        self.depth_img_dirs        = self.get_files(depth_dir)
        self.masked_depth_img_dirs = self.get_files(masked_depth_dir)
        self.skeleton_files        = self.get_files(skeleton_dir)
        # Check if metadata saved to disk
        self.skip_load             = False
        self.metadata              = self.check_metadata()




    def load(self):
        '''
        Long running load function - loads all metadata: video info, rotation,
        translation, and scale vectors
        '''
        if self.skip_load:
            return

        # Metadata
        self.metadata = []
        for vid_idx in tqdm(range(self.num_vids), "Getting video info"):
            self.metadata.append(self.get_metadata(vid_idx))

        # Rotation & Translation matrix
        vid_sets = set([m['video_set'] for m in self.metadata])
        rot_trans = {}
        for vid_set in tqdm(vid_sets, "Calculating rotation & translation matrices"):
            rot_trans[vid_set] = self.get_rotation_translation(vid_set)

        # Scale factor
        scale_rot_trans = {}
        for m in tqdm(self.metadata, "Calculating scale factor for offset"):
            idx = m['video_index']
            min_loss = 10000
            for R, T in rot_trans[m['video_set']]:
                SRT = self.get_scale(idx, R, T)
                # Use previous scale if this video has no skeleton data
                if SRT is None:
                    scale_rot_trans[idx] = scale_rot_trans[idx - 1]
                    break
                if SRT[3] < min_loss:
                    min_loss = SRT[3]
                    scale_rot_trans[idx] = SRT

        # Put all info in metadata file
        for idx, m in enumerate(self.metadata):
            self.metadata[idx]['scale']  = scale_rot_trans[m['video_index']][0]
            self.metadata[idx]['R']      = scale_rot_trans[m['video_index']][1]
            self.metadata[idx]['T']      = scale_rot_trans[m['video_index']][2]
            self.metadata[idx]['s_loss'] = scale_rot_trans[m['video_index']][3]


        # Pickle metadata for later
        pickle.dump(self.metadata, open('cache/metadata.pkl', 'wb'))




    def check_metadata(self):
        '''
        Checks current path for saved metadata and prompts user to use this
        or reload
        '''
        if 'metadata.pkl' in os.listdir(os.path.join(os.curdir, 'cache')):
            if yesno('metadata.pkl found in cache directory. Use this file?'):
                self.skip_load = True
                return pickle.load(open('cache/metadata.pkl', 'rb'))
        else:
            print('cache/metadata.pkl not found. Use load() to load metadata')




    def check_cache(self, item_type, vid_id):
        '''
        Checks the cache folder for the item requested

        Returns
        -------
        cached_item : The requested item (if found) or False (if not found)
        '''
        cached_files = os.listdir(os.path.join(os.curdir, 'cache', item_type))
        file_prefix = os.path.join(os.curdir, 'cache', item_type, '{:05}'.format(vid_id))

        # Pickle
        if '{:05}.pickle'.format(vid_id) in cached_files:
            print("Loading {}/{:05}.pickle from cache".format(item_type, vid_id))
            return pickle.load(open(file_prefix + '.pickle', 'rb'))

        # Numpy (uncompressed)
        elif '{:05}.npy'.format(vid_id) in cached_files:
            print("Loading {}/{:05}.npy from cache".format(item_type, vid_id))
            return np.load(file_prefix + '.npy')

        # Numpy (compressed)
        elif '{:05}.npz'.format(vid_id) in cached_files:
            print("Loading {}/{:05}.npz from cache".format(item_type, vid_id))
            return np.load(file_prefix + '.npz')

        # Not found
        else:
            return False





    def cache(self, item, item_type, vid_id, compress=False):
        '''
        Saves the item in the cache folder
        '''
        print("Caching {}/{:05}".format(item_type, vid_id))

        # Item already cached
        cached_files = os.listdir(os.path.join(os.curdir, 'cache', item_type))
        cached_vid_ids = [os.path.splitext(path)[0] for path in cached_files]
        if '{:05}'.format(vid_id) in cached_vid_ids:
            print("{}/{:05} already cached, skipping".format(item_type, vid_id))
            return

        file_prefix = os.path.join(os.curdir, 'cache', item_type, '{:05}'.format(vid_id))

        # Use numpy save if a numpy object
        if type(item) == np.ndarray:
            if compress:
                np.savez_compressed(file_prefix + '.npz', item)
            else:
                np.save(file_prefix + '.npy', item)

        # Use pickle on all other objects
        else:
            pickle.dump(item, open(file_prefix + '.pickle', 'wb'))





    def get_files(self, folder):
        '''
        Helper function for finding file names
        '''
        return sorted(glob.glob(os.path.join(folder, "*")))




    def get_metadata(self, vid_id):
        '''
        Get the metadata for a specified video
        '''
        vid_name = self.rgb_vids[vid_id]
        match = re.match(compiled_regex, vid_name)
        setup, camera, performer, replication, action = [*map(int, match.groups())]
        return {
            'video_index': vid_id,
            'video_set':   (setup, camera),
            'setup':       setup,
            'camera':      camera,
            'performer':   performer,
            'replication': replication,
            'action':      action,
            'num_frames':  len(self.get_files(self.masked_depth_img_dirs[vid_id])),
        }




    def get_rgb_vid_images(self, vid_id, grayscale=False):
        '''
        Returns the rgb frames for a specified video in a ndarray of size:
            [video_frames * 1080 * 1920 * 3]
        '''
        vid = av.open(self.rgb_vids[vid_id])
        imgs = []
        for packet in vid.demux():
            for frame in packet.decode():
                if not grayscale:
                    imgs.append(frame.to_rgb().to_nd_array())
                else:
                    imgs.append(np.array(frame.to_image().convert('L')))
        return np.stack(imgs)




    def get_ir_vid_images(self, vid_id):
        '''
        Returns the ir frames for a specified video in a ndarray of size:
            [video_frames * 424 * 512]
        '''
        vid = av.open(self.ir_vids[vid_id])
        imgs = []
        for packet in vid.demux(vid.streams[0]):
            frame = packet.decode_one()
            if frame is None:
                break
            imgs.append(np.array(frame.to_image().convert('L')))
        return np.stack(imgs)




    def get_depth_images(self, vid_id):
        '''
        Returns the depth frames (in mm) for a specified video in a ndarray of size:
            [video_frames * 424 * 512]
        '''
        depth_image_paths = self.get_files(self.masked_depth_img_dirs[vid_id])
        imgs = [cv2.imread(img_path, -1) for img_path in depth_image_paths]
        return np.stack(imgs)




    def get_point_clouds(self, vid_id):
        '''
        Returns the point clouds for a specified video in an ndarray of size:
            [video_frames * N * 3]
         where N is the maximum number of points and the 3rd dimension is the
         (x,y,z) coordinate.

        The frames that don't have N points will be 0 padded.
        '''
        point_clouds = []
        for depth_img in self.get_depth_images(vid_id):
            point_clouds.append(self.depth_to_pc(depth_img))
        return np.stack(point_clouds)





    def get_2D_optical_flow(self, vid_id):
        '''
        Returns the 2D optical flow vectors for a video in an ndarray of size:
            [video_frames - 1 * 2 * vid_height * vid_width]
        '''
        vid = self.get_rgb_vid_images(vid_id, True)
        prev_frame = vid[0]
        flow = None
        flow_maps = np.zeros([len(vid) - 1, 2, vid.shape[1], vid.shape[2]])
        for kk in range(1,len(vid)-1):
            flow = cv2.calcOpticalFlowFarneback(vid[kk-1], vid[kk], flow, 0.4,
                1, 15, 3, 8, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow_maps[kk-1,0,:,:] = flow[:,:,0].copy()
            flow_maps[kk-1,1,:,:] = flow[:,:,1].copy()
        return flow_maps



    def get_rgb_3D_maps(self, vid_id):
        '''
        Creates a map from rgb pixels to the depth camera xyz coordinate at that
        pixel.

        Returns
        -------
        rgb_xyz : ndarray, shape [video_frames * 1080 * 1920 * 3]
        '''

        # Get metadata - for rotation and translation matrices
        for metadatum in self.metadata:
            if metadatum['video_index'] == vid_id:
                m = metadatum
                break

        # Get depth images
        depth_ims = self.get_depth_images(vid_id)
        depth_ims = depth_ims.astype(np.float32)/1000.0
        depth_ims[depth_ims == 0] = -1000

        # Constants - image size
        frames, H_depth, W_depth = depth_ims.shape
        W_rgb, H_rgb = 1920, 1080

        # Depth --> Depth-camera coordinates
        Y, X = np.mgrid[0:H_depth, 0:W_depth]
        x_3D = (X - cx_d) * depth_ims / fx_d
        y_3D = (Y - cy_d) * depth_ims / fy_d

        # Apply rotation and translation
        xyz_d = np.stack([x_3D, y_3D, depth_ims], axis=3)
        xyz_rgb = m['T']*m['scale'] + m['R'] @ xyz_d[:,:,:,:,np.newaxis]

        # RGB-camera coordinates --> RGB pixel coordinates
        x_rgb = (xyz_rgb[:,:,:,0] * rgb_mat[0,0] / xyz_rgb[:,:,:,2]) + rgb_mat[0,2]
        y_rgb = (xyz_rgb[:,:,:,1] * rgb_mat[1,1] / xyz_rgb[:,:,:,2]) + rgb_mat[1,2]
        x_rgb[x_rgb >= W_rgb] = 0
        y_rgb[y_rgb >= H_rgb] = 0

        # Fill in sparse array
        rgb_xyz_sparse = np.zeros([frames, H_rgb, W_rgb, 3])
        for frame in range(frames):
            for y in range(H_depth):
                for x in range(W_depth):
                    rgb_xyz_sparse[frame, int(y_rgb[frame,y,x]), int(x_rgb[frame,y,x])] = xyz_d[frame, y, x]

        # Fill in the rest of the sparse matrix
        invalid = (rgb_xyz_sparse == 0)
        ind = scipy.ndimage.distance_transform_edt(invalid,
                                                   return_distances=False,
                                                   return_indices=True)
        rgb_xyz = rgb_xyz_sparse[tuple(ind)]

        # Remove background values by zeroing them out
        rgb_xyz[rgb_xyz[:,:,:,2] < 0] = 0

        return rgb_xyz






    def get_3D_optical_flow(self, vid_id, _rgb_3D=None, _op_flow_2D=None):
        '''
        Turns 2D optical flow into 3D

        Returns
        -------
        op_flow_3D : list (length = #frames in video-1) of ndarrays
            (shape: optical_flow_arrows * 6) (6 --> x,y,z,dx,dy,dz)
        '''

        # Get rgb to 3D map and the 2D rgb optical flow vectors
        start = dt.datetime.now()
        print("Getting rgb 3D map.", end="")
        rgb_3D = _rgb_3D if _rgb_3D is not None else self.get_rgb_3D_maps(vid_id)
        print(" Done {}".format(dt.datetime.now() - start))
        start = dt.datetime.now()
        print("Getting 2D optical flow.", end="")
        op_flow_2D = _op_flow_2D if _op_flow_2D is not None else self.get_2D_optical_flow(vid_id)
        print(" Done {}".format(dt.datetime.now() - start))
        start = dt.datetime.now()
        print("Building 3D optical flow.", end="")

        # Build list of framewise 3D optical flow vectors
        op_flow_3D = []

        # Note: starting at frame 2 (flow maps start at previous frame)
        for frame in range(1, op_flow_2D.shape[0]):
            # Only look at non-zero rgb points
            rgb_nonzero = np.nonzero(rgb_3D[frame,:,:,2])
            flow_vectors = []

            for u, v in zip(rgb_nonzero[1], rgb_nonzero[0]):
                # Get optical flow vector
                du, dv = op_flow_2D[frame, :, v, u]

                # Get start and end position in 3D using the flow map vector
                p0 = rgb_3D[frame - 1, int(v - dv), int(u - du)]
                if p0[2] == 0: continue # Only want vectors that started at a non-zero point
                p1 = rgb_3D[frame, v, u]

                # Get change in position if optical flow vector is large enough
                if np.linalg.norm([du,dv]) < 1.0:
                    dp = np.array([0,0,0])
                else:
                    dp = p1 - p0

                flow_vectors.append(np.concatenate([p0, dp]))

            # Stack list of flow vectors into one array
            op_flow_3D.append(np.stack(flow_vectors))

        # Zero mean x y & z (the starting point)
        all_vecs = np.concatenate(op_flow_3D)
        m = np.mean(all_vecs, axis=0)

        for frame in range(len(op_flow_3D)):
            op_flow_3D[frame][:,0] -= m[0]
            op_flow_3D[frame][:,1] -= m[1]
            op_flow_3D[frame][:,2] -= m[2]

        print(" Done {}".format(dt.datetime.now() - start))
        return op_flow_3D




    def get_voxel_flow(self, vid_id, cache=False):
        '''
        Get voxelized 3D optical flow tensor (sparse)

        Returns
        -------
        voxel_flow : ndarray, shape [video frames,  100,  100,  100,  4]
            3 is for dx, dy, dz
        '''
        VOXEL_SIZE = 100

        # Check cache for voxel flow
        voxel_flow = self.check_cache('voxel_flow', vid_id)
        if voxel_flow != False:
            return voxel_flow

        # Check cache for 3D optical flow
        op_flow_3D = self.check_cache('optical_flow_3D', vid_id)
        if op_flow_3D == False:
            op_flow_3D = self.get_3D_optical_flow(vid_id)

        # Pull useful stats out of optical flow
        num_frames = len(op_flow_3D)
        all_xyz = np.array([flow[0:3] for frame in op_flow_3D for flow in frame])
        max_x, max_y, max_z = np.max(all_xyz, axis=0) + 0.00001
        min_x, min_y, min_z = np.min(all_xyz, axis=0)

        voxel_flow_dicts = []
        # Backwards fill-in of voxel grid
        for frame in tqdm(range(num_frames), "Filling in Voxel Grid"):

            # Interpolate and discretize location of the voxels in the grid
            vox_x = np.floor((op_flow_3D[frame][:,0] - min_x)/(max_x - min_x) * VOXEL_SIZE).astype(int)
            vox_y = np.floor((op_flow_3D[frame][:,1] - min_y)/(max_y - min_y) * VOXEL_SIZE).astype(int)
            vox_z = np.floor((op_flow_3D[frame][:,2] - min_z)/(max_z - min_z) * VOXEL_SIZE).astype(int)

            # Get unique tuples of voxels, then average the flow vectors at each voxel
            filled_voxels = set([(a,b,c) for a,b,c in np.stack([vox_x,vox_y,vox_z]).T])

            # Get the average displacement vector in each voxel
            num_disp_vecs = {tup: 0.0 for tup in filled_voxels}
            sum_disp_vecs = {tup: np.zeros([3]) for tup in filled_voxels}
            avg_disp_vecs = {tup: np.zeros([3]) for tup in filled_voxels}
            for i in range(len(op_flow_3D[frame])):
                num_disp_vecs[(vox_x[i], vox_y[i], vox_z[i])] += 1.0
                sum_disp_vecs[(vox_x[i], vox_y[i], vox_z[i])] += op_flow_3D[frame][i,3:]

            for tup in filled_voxels:
                avg_disp_vecs[tup] = sum_disp_vecs[tup]/num_disp_vecs[tup]

            voxel_flow_dicts.append(avg_disp_vecs)

        # Turn voxel flow into numpy tensor
        voxel_flow_tensor = np.zeros([num_frames, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 4])
        for frame in range(num_frames):
            for vox, disp_vec in voxel_flow_dicts[frame].items():
                voxel_flow_tensor[frame, vox[0], vox[1], vox[2], 0] = 1.0
                voxel_flow_tensor[frame, vox[0], vox[1], vox[2], 1:] = disp_vec

        # Cache the voxel flow
        if cache:
            self.cache(voxel_flow_tensor, 'voxel_flow', vid_id, compress=True)

        return voxel_flow_tensor





    def get_rotation_translation(self, vid_set):
        '''
        Calculate the rotation and translation vectors given the unique id
        for a set of videos
        '''

        r_t_pairs = []
        attempts = 20
        for _ in range(attempts):
            # Build dataframe with some random set of data in the video set
            vid_set_dfs = []
            for m in self.metadata:
                if m['video_set'] == vid_set and np.random.rand() < 0.5:
                    vid_set_dfs.append(self.get_skeleton_data(m['video_index']))
            full_df = pd.concat(vid_set_dfs, ignore_index=True)

            rgb = np.array([full_df['color']]).astype('float32')
            d   = np.array([full_df['depth']]).astype('float32')

            F, mask = cv2.findFundamentalMat(d, rgb)    # Fundamental matrix
            E = rgb_mat.T @ F @ d_mat                   # Essential matrix
            R1, R2, T = cv2.decomposeEssentialMat(E)    # Decompose essential matrix

            # Get rotation matrix that looks most similar to the identity matrix
            R = R1 if abs(np.sum(R1 - np.identity(3))) < abs(np.sum(R2 - np.identity(3))) else R2

            r_t_pairs.append((R, T))
        return r_t_pairs




    def get_scale(self, vid_id, R, T):
        '''
        Calculate the optimal scale factor for a given video
        '''
        skeleton_df = self.get_skeleton_data(vid_id)

        if len(skeleton_df) == 0: # One of the empty skeleton files
            return None

        loc = np.array([skeleton_df['loc']]).astype('float32')
        rgb = np.array([skeleton_df['color']]).astype('float32')
        d   = np.array([skeleton_df['depth']]).astype('float32')

        # Get the 3D depth data
        d_3D = np.linalg.inv(d_mat) @ np.array([d[0][3][0], d[0][3][1], 1]) * loc[0][3][2]

        # Loss function to minimize
        def loss_func(s):
            rgb_3D = (R @ d_3D.reshape(1,-1).T) + T*s        # Convert depth 3D to rgb 3D
            c_2D = ((rgb_mat @ rgb_3D) / rgb_3D[2]).T[0][:2] # Convert to 2D
            return np.linalg.norm(c_2D.T - rgb[0][3])

        opt_dict = scipy.optimize.minimize(loss_func, 0)
        loss = opt_dict['fun']
        scale = opt_dict['x'][0]
        return (scale, R, T, loss)




    def depth_to_pc(self, depth_img):
        '''
        Convert a Kinect depth image to a point cloud.  Input is a depth image file.
        Output is an [N x 3] matrix of 3D points in units of meters.

        (The point cloud is normalized to zero mean and to fit within the unit circle)
        '''

        imgDepthAbs = depth_img.astype(np.float32)/1000.0  # convert to meters

        H = imgDepthAbs.shape[0]
        W = imgDepthAbs.shape[1]

        xx, yy, = np.meshgrid(range(0,W), range(0,H))

        # Just collect valid points
        valid = imgDepthAbs > 0
        xx = xx[valid]
        yy = yy[valid]
        imgDepthAbs = imgDepthAbs[valid]

        # Project all depth pixels into 3D (in the depth camera's coordinate system)
        X = (xx - cx_d) * imgDepthAbs / fx_d
        Y = (yy - cy_d) * imgDepthAbs / fy_d
        Z = imgDepthAbs

        # Create an [npoints * 3] ndarray of depth values
        points3d = np.array([X.flatten(), Y.flatten(), Z.flatten()])

        # Zero mean the points
        m = np.mean(points3d, axis=0)
        points3d[:,0] -= m[0]
        points3d[:,1] -= m[1]
        points3d[:,2] -= m[2]

        # Now get the point with the maximum length and normalize each point to that length
        # so all points fit within the unit sphere.
        max_vec_len = np.linalg.norm(points3d, axis=1).max()
        points3d /= max_vec_len

        return points3d




    def get_skeleton_data(self, vid_id):
        from collections import namedtuple
        JointList = [
            'spine_base', 'spine_middle', 'neck', 'head', 'shoulder_left', 'elbow_left',
            'wrist_left', 'hand_left', 'shoulder_right', 'elbow_right', 'wrist_right',
            'hand_right', 'hip_left', 'knee_left', 'ankle_left', 'foot_left',
            'hip_right', 'knee_right', 'ankle_right', 'foot_right', 'spine',
            'left_hand_tip', 'thumb_left', 'right_hand_tip', 'thumb_right',
        ]
        JointIndexMap = {idx:j for idx,j in enumerate(JointList)}
        FrameData = namedtuple('FrameData', 'loc depth color joint_name \
                               joint_index person_index frame_index video_index')

        # Read skeleton file data
        with open(self.skeleton_files[vid_id], 'r') as f:
            data = f.readlines()

        # Pull out useful data from file
        joint_data = []
        for frame_idx in range(int(data.pop(0))):
            for body_idx in range(int(data.pop(0))):
                body = data.pop(0)
                for joint_idx in range(int(data.pop(0))):
                    line  = data.pop(0).split()
                    joint_data.append((frame_idx, body_idx, joint_idx, line[:7]))

        # Convert to known format
        joints = []
        for joint in joint_data:
            x = np.array(joint[3], dtype=np.float32)
            joint = FrameData(
                x[:3], x[3:5], x[5:7], JointIndexMap[joint[2]], joint[2],
                joint[1], joint[0], vid_id
            )
            joints.append(joint)

        skeleton_df = pd.DataFrame(joints)
        return skeleton_df


if __name__ == '__main__':
    dataset = NTU()
    for vid in range(0,60):
        print("Video {}".format(vid))
        # op_flow_3D = dataset.get_3D_optical_flow(vid)
        voxel_flow = dataset.get_voxel_flow(vid, cache=True)
