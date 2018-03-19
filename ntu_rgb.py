# TODO
# - Better docs -- use sphinx (or something)
# - Turn this into an installable python module


'''
NTU RGB+D Action Recognition Dataset helper

website: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
github: https://github.com/shahroudy/NTURGB-D
'''
# from __future__ import division
import sys, os, glob
import numpy as np
import pandas as pd
import cv2
import av
import scipy, scipy.optimize, scipy.ndimage
# import line_profiler
import datetime as dt
import re
from tqdm import tqdm
from prompter import yesno
import pickle
from progress_meter import ProgressMeter

from config import *



##################################################
# Regex for pulling metadata out of video name
compiled_regex = re.compile('.*S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3}).*')



##################################################
# Subject ids used in original paper for training
TRAIN_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAIN_VALID_IDS = ([1, 2, 5, 8, 9, 13, 14, 15, 16, 18, 19, 27, 28, 31, 34, 38], [4, 17, 25, 35])



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





class NTU():
    def __init__(self):
        self.num_vids              = len(self.get_files(CACHE_RGB_VID))
        self.rgb_vids              = self.get_files(CACHE_RGB_VID)
        self.ir_vids               = self.get_files(CACHE_IR_VID)
        self.depth_img_dirs        = self.get_files(CACHE_DEPTH)
        self.masked_depth_img_dirs = self.get_files(CACHE_MASKED_DEPTH)
        self.skeleton_files        = self.get_files(CACHE_SKELETONS)

        # Check if metadata saved to disk
        self.metadata              = self.check_metadata()

        # Set train test splits
        self.set_splits()

        # Video labels
        self.id_to_action = list(pd.DataFrame(self.metadata)['action'] - 1)




    def load_metadata(self):
        '''
        Long running load function - loads all metadata: video info, rotation,
        translation, and scale vectors
        '''
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
        pickle.dump(self.metadata, open(CACHE_METADATA, 'wb'))

        return self.metadata





    def check_metadata(self):
        '''
        Checks cache for metadata

        Returns
        -------
        metadata : A list of dicts with information about the videos
        '''
        # TODO: remove
        return pickle.load(open(CACHE_METADATA, 'rb'))
        ##############

        if os.path.isfile(CACHE_METADATA):
            if yesno('metadata.pickle found in cache. Use this file?'):
                self.skip_load = True
                return pickle.load(CACHE_METADATA, 'rb')

        if yesno('Cached metadata not found. Want to create it now?'):
            return self.load_metadata()
        else:
            exit()




    def set_splits(self):
        '''
        Sets the train/test splits

        Cross-Subject Evaluation:
            Train ids = 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
                        28, 31, 34, 35, 38

        Cross-View Evaluation:
            Train camera views: 2, 3
        '''
        # Save the dataset as a dataframe
        dataset = pd.DataFrame(self.metadata)

        # Get the train split ids
        train_ids_camera  = [2, 3]

        # Cross-Subject splits
        self.train_split_subject = list(
            dataset[dataset.performer.isin(TRAIN_IDS)]['video_index'])
        self.train_split_subject_with_validation = list(
            dataset[dataset.performer.isin(TRAIN_VALID_IDS[0])]['video_index'])
        self.validation_split_subject = list(
            dataset[dataset.performer.isin(TRAIN_VALID_IDS[1])]['video_index'])
        self.test_split_subject = list(
            dataset[~dataset.performer.isin(TRAIN_IDS)]['video_index'])

        # Cross-View splits
        self.train_split_camera  = list(
            dataset[dataset.camera.isin(train_ids_camera)]['video_index'])
        self.test_split_camera  = list(
            dataset[~dataset.camera.isin(train_ids_camera)]['video_index'])





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
        flow = None
        op_flow_2D = np.zeros([len(vid) - 1, 2, vid.shape[1], vid.shape[2]]).astype(np.float32)
        for kk in tqdm(range(1,len(vid)), "Building 2D optical flow tensor"):
            flow = cv2.calcOpticalFlowFarneback(vid[kk-1], vid[kk], flow, 0.4,
                1, 15, 3, 8, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            op_flow_2D[kk-1,0,:,:] = flow[:,:,0].copy()
            op_flow_2D[kk-1,1,:,:] = flow[:,:,1].copy()
        return op_flow_2D





    def get_rgb_3D_maps(self, vid_id):
        '''
        Creates a map from rgb pixels to the depth camera xyz coordinate at that
        pixel.

        Returns
        -------
        rgb_xyz : ndarray, shape [video_frames * 1080 * 1920 * 3]
        '''

        # Get depth images
        depth_ims = self.get_depth_images(vid_id)
        depth_ims = depth_ims.astype(np.float32)/1000.0

        # Make background negative so can discriminate between background
        # and empty values
        depth_ims[depth_ims == 0] = -1000

        # Constants - image size
        frames, H_depth, W_depth = depth_ims.shape
        W_rgb, H_rgb = 1920, 1080

        # Depth --> Depth-camera coordinates
        Y, X = np.mgrid[0:H_depth, 0:W_depth]
        x_3D = (X - cx_d) * depth_ims / fx_d
        y_3D = (Y - cy_d) * depth_ims / fy_d

        # Get metadata - for rotation and translation matrices
        m = next((meta for meta in self.metadata if meta['video_index'] == vid_id), None)

        # Apply rotation and translation
        xyz_d = np.stack([x_3D, y_3D, depth_ims], axis=3)
        xyz_rgb = m['T']*m['scale'] + m['R'] @ xyz_d[:,:,:,:,np.newaxis]

        # RGB-camera coordinates --> RGB pixel coordinates
        x_rgb = (xyz_rgb[:,:,:,0] * rgb_mat[0,0] / xyz_rgb[:,:,:,2]) + rgb_mat[0,2]
        y_rgb = (xyz_rgb[:,:,:,1] * rgb_mat[1,1] / xyz_rgb[:,:,:,2]) + rgb_mat[1,2]

        # Convert index arrays to integer and clip to rgb dimensions
        x_rgb = np.clip(x_rgb, 0, W_rgb-1).astype(int)
        y_rgb = np.clip(y_rgb, 0, H_rgb-1).astype(int)

        # Build rgb 3D coordinate tensor
        rgb_xyz = np.zeros([frames, H_rgb, W_rgb, 3]).astype(np.float32)
        for frame in tqdm(range(frames),
                          "Building rgb 3D-coordinate tensor {}".format(vid_id)):
            # Fill tensor with sparse values
            rgb_xyz[frame, y_rgb[frame,:,:,0], x_rgb[frame,:,:,0]] = xyz_d[frame]

            # Interpolate to fill in the rest of the tensor
            empty = (rgb_xyz[frame] == 0)
            ind = scipy.ndimage.distance_transform_edt(empty,
                                                       return_distances=False,
                                                       return_indices=True)

            # Set the values
            rgb_xyz[frame] = rgb_xyz[frame, ind[0], ind[1], ind[2]]

            # Remove background values by zeroing them out
            rgb_xyz[frame, rgb_xyz[frame,:,:,2] < 0] = 0

        return rgb_xyz





    def get_3D_optical_flow(self, vid_id, cache=False):
        '''
        Turns 2D optical flow into 3D

        Returns
        -------
        op_flow_3D : list (length = #frames in video-1) of ndarrays
            (shape: optical_flow_arrows * 6) (6 --> x,y,z,dx,dy,dz)
        '''

        # Check cache for optical flow
        if os.path.isfile(CACHE_3D_OP_FLOW + '/{:05}.npz'.format(vid_id)):
            # print("Found 3D optical flow {:05} in cache".format(vid_id))
            return np.load(CACHE_3D_OP_FLOW + '/{:05}.npz'.format(vid_id))['arr_0']

        # Get rgb to 3D map and the 2D rgb optical flow vectors
        rgb_xyz = self.get_rgb_3D_maps(vid_id)
        op_flow_2D = self.get_2D_optical_flow(vid_id)

        # Build list of framewise 3D optical flow vectors
        op_flow_3D = []

        # Note: 2D optical flow goes from frame t to t+1
        for frame in tqdm(range(op_flow_2D.shape[0]),
                          "Building 3D optical flow tensor {}".format(vid_id)):
            # Get the points in frame t+1
            p1 = np.nonzero(rgb_xyz[frame+1,:,:,2])

            # Get the starting vector p0 from frame t
            dudv = op_flow_2D[frame, :, p1[0], p1[1]][:,:]
            p0u = p1[1] - dudv[:,0]
            p0v = p1[0] - dudv[:,1]

            # Clip to pixels
            p0v = np.clip(p0v, 0, 1079).astype(int)
            p0u = np.clip(p0u, 0, 1919).astype(int)

            # Get the points in frame t
            p0 = rgb_xyz[frame, p0v, p0u]

            # Get the displacement vector between p(t) and p(t+1)
            disp_vecs = rgb_xyz[frame+1, p1[0], p1[1]] - p0
            disp_vecs[np.sum(dudv*dudv, axis=1) < 0.5] = np.array([0,0,0])

            # Combine start (x,y,z) and displacement (dx,dy,dz) into ndarray
            start_disp = np.hstack([p0, disp_vecs])

            # Get rid of duplicates
            start_disp = np.unique(start_disp, axis=0)

            # Add to list
            op_flow_3D.append(start_disp[start_disp[:,2] != 0])

        # Zero mean x y & z (the starting point)
        m = np.mean(np.concatenate(op_flow_3D), axis=0)
        for frame in op_flow_3D:
            frame[:,:3] -= m[:3]

        # Pad and concatenate the vectors
        lens = np.array([len(i) for i in op_flow_3D])
        op_flow_3D_vec = np.zeros([len(op_flow_3D), max(lens), 6])
        for idx,frame in enumerate(op_flow_3D):
            op_flow_3D_vec[idx,:lens[idx]] += frame

        if cache:
            np.savez_compressed(CACHE_3D_OP_FLOW + '/{:05}'.format(vid_id), op_flow_3D_vec)

        return op_flow_3D




    def get_voxel_flow(self, vid_id, cache=False):
        '''
        Voxelize the 3D optical flow tensor (sparse)

        Parameters
        ----------
        vid_id : int
            The video to get the voxel flow for. {1-56879}
        cache : bool, optional
            Whether to cache the voxel flow after creating it.

        Returns
        -------
        voxel_flow_tensor : ndarray
            The voxel flow for the video, shape [frames,  4,  100,  100,  100].
            The 4 values in the 2nd dimension are 0: {0,1} indicating voxel is
            filled, 1-3: dx, dy, dz components of average optical flow at that
            voxel
        '''
        VOXEL_SIZE = 100

        # Get 3D optical flow
        op_flow_3D = self.get_3D_optical_flow(vid_id)

        # Pull useful stats out of optical flow
        num_frames = len(op_flow_3D)
        all_xyz = np.vstack(op_flow_3D)
        max_x, max_y, max_z = np.max(all_xyz, axis=0)[:3] + 0.00001
        min_x, min_y, min_z = np.min(all_xyz, axis=0)[:3]

        # Fill in the voxel grid
        voxel_flow_tensor = np.zeros([num_frames, 4, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])
        for frame in range(num_frames):#tqdm(range(num_frames), "Filling in Voxel Grid"):

            # Interpolate and discretize location of the voxels in the grid
            vox_x = np.floor((op_flow_3D[frame][:,0] - min_x)/(max_x - min_x) * VOXEL_SIZE).astype(np.uint8)
            vox_y = np.floor((op_flow_3D[frame][:,1] - min_y)/(max_y - min_y) * VOXEL_SIZE).astype(np.uint8)
            vox_z = np.floor((op_flow_3D[frame][:,2] - min_z)/(max_z - min_z) * VOXEL_SIZE).astype(np.uint8)

            # Add all interpolated values to the correct location in the tensor
            np.add.at(voxel_flow_tensor, (frame, 0, vox_x, vox_y, vox_z), 1)
            np.add.at(voxel_flow_tensor, (frame, 1, vox_x, vox_y, vox_z), op_flow_3D[frame][:,3])
            np.add.at(voxel_flow_tensor, (frame, 2, vox_x, vox_y, vox_z), op_flow_3D[frame][:,4])
            np.add.at(voxel_flow_tensor, (frame, 3, vox_x, vox_y, vox_z), op_flow_3D[frame][:,5])

            # Average values
            voxel_flow_tensor[frame, 1, vox_x, vox_y, vox_z] /= voxel_flow_tensor[frame, 0, vox_x, vox_y, vox_z]
            voxel_flow_tensor[frame, 2, vox_x, vox_y, vox_z] /= voxel_flow_tensor[frame, 0, vox_x, vox_y, vox_z]
            voxel_flow_tensor[frame, 3, vox_x, vox_y, vox_z] /= voxel_flow_tensor[frame, 0, vox_x, vox_y, vox_z]
            voxel_flow_tensor[frame, 0, vox_x, vox_y, vox_z] = 1

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





    def get_rgb_mask(self, vid_id):
        '''
        Get a bounding box around where the action should be in the rgb image

        Parameters
        ----------
        vid_id : int
            The video id

        Returns
        -------
        bbox : ndarray
            The bounding box, [y min, x min, y max, x max] (min being top left)
        '''
        # Get bounding box for depth image
        depth_paths = self.get_files(self.masked_depth_img_dirs[vid_id])
        depth_im    = cv2.imread(depth_paths[0], -1)
        depth_nz    = np.nonzero(depth_im)
        depth_bbox  = np.concatenate([np.min(depth_nz, 1), np.max(depth_nz, 1)])

        # Get depth pixel to rgb pixel map
        depth_im = depth_im.astype(np.float32)/1000.0
        depth_im[depth_im == 0] = -1000
        Y, X = np.mgrid[0:424, 0:512]
        x_3D = (X - cx_d) * depth_im / fx_d
        y_3D = (Y - cy_d) * depth_im / fy_d
        m = next((meta for meta in self.metadata if meta['video_index'] == vid_id), None)
        xyz_d = np.stack([x_3D, y_3D, depth_im], axis=2)
        xyz_rgb = m['T']*m['scale'] + m['R'] @ xyz_d[:,:,:,np.newaxis]
        xyz_rgb = xyz_rgb.squeeze(axis=3)
        x_rgb = (xyz_rgb[:,:,0] * rgb_mat[0,0] / xyz_rgb[:,:,2]) + rgb_mat[0,2]
        y_rgb = (xyz_rgb[:,:,1] * rgb_mat[1,1] / xyz_rgb[:,:,2]) + rgb_mat[1,2]
        x_rgb = np.clip(x_rgb, 0, 1919).astype(int)
        y_rgb = np.clip(y_rgb, 0, 1079).astype(int)
        d_2_rgb = np.stack([y_rgb, x_rgb], axis=2)

        # Get bounding box in rgb
        rgb_mins = d_2_rgb[depth_bbox[0], depth_bbox[1]]
        rgb_maxs = d_2_rgb[depth_bbox[2], depth_bbox[3]]
        rgb_bbox = np.concatenate([rgb_mins, rgb_maxs])

        return rgb_bbox






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



def create_voxel_flows():
    dataset = NTU()
    for vid in range(10):
        dataset.get_voxel_flow(vid, cache=True)


def create_all_voxel_flows():
    dataset = NTU()
    for vid in range(5000):
        start = dt.datetime.now()
        dataset.get_voxel_flow(vid, cache=True)
        print("Total time: {}".format(dt.datetime.now() - start))


def create_all_3D_op_flows():
    dataset = NTU()
    step = int(sys.argv[2])
    for vid in range((int(sys.argv[1])-1)*step, min(int(sys.argv[1])*step, dataset.num_vids)):
        start = dt.datetime.now()
        dataset.get_3D_optical_flow(vid, cache=True)
        print("Total time: {}".format(dt.datetime.now() - start))

if __name__ == '__main__':
    dataset = NTU()
    x = dataset.get_
