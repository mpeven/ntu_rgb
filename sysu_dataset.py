import os, sys, glob
import cv2
from tqdm import tqdm
import numpy as np
import scipy, scipy.ndimage
from opengl_viewer.opengl_viewer import OpenGlViewer


# DATASET_LOCATION = '/Users/mpeven/Downloads/SYSU3DAction_2'
DATASET_LOCATION = '/home/mike/Documents/SYSU'
op_flow_3D_dir = '/hdd/Datasets/SYSU/op_flow_3D'


class SYSU:
    def __init__(self):
        self.dataset = self.get_files()
        self.num_vids = len(self.dataset)

    def check_files(self):
        for subject in sorted(glob.glob(DATASET_LOCATION + '/*')):
            if os.path.basename(subject) == 'matlabcode': continue
            videos = glob.glob(subject + '/*')
            for video in sorted(videos):
                dl = len(glob.glob(video + '/depth/*'))
                rl = len(glob.glob(video + '/rgb/*'))
                if dl != rl or dl == 0:
                    print("Error: {} - {}".format(os.path.basename(subject), os.path.basename(video)))

    def get_files(self):
        all_videos = []
        for subject in sorted(glob.glob(DATASET_LOCATION + '/*')):
            if os.path.basename(subject) == 'matlabcode': continue
            videos = glob.glob(subject + '/*')
            for video in sorted(videos):
                depth_files = sorted(glob.glob(video + '/depth/*'))
                rgb_files = sorted(glob.glob(video + '/rgb/*'))
                if len(depth_files) == len(rgb_files) and len(depth_files) != 0:
                    all_videos.append({
                        'subject': os.path.basename(subject),
                        'video': int(os.path.basename(video.replace('video', ''))),
                        'depth_files': depth_files,
                        'rgb_files': rgb_files
                    })
        return all_videos

    def get_rgb_vid_images(self, vid_id, grayscale=False):
        all_ims = []
        for im in self.dataset[vid_id]['rgb_files']:
            if not grayscale:
                all_ims.append(cv2.imread(im))
            else:
                all_ims.append(cv2.imread(im, 0))
        return np.stack(all_ims)

    def get_depth_images(self, vid_id):
        all_ims = []
        for im in self.dataset[vid_id]['depth_files']:
            all_ims.append(cv2.resize(cv2.imread(im, -1), (640,480)))
        return np.stack(all_ims)

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
        depth_ims = self.get_depth_images(vid_id)
        depth_ims = depth_ims.astype(np.float32)/10000.0
        depth_ims[depth_ims > 2.5] = 0
        frames, H_depth, W_depth = depth_ims.shape
        W_rgb, H_rgb = 640, 480
        rgb_xyz = np.zeros([frames, H_rgb, W_rgb, 3]).astype(np.float32)
        Y, X = np.mgrid[0:H_depth, 0:W_depth]
        rgb_xyz[:, :, :, 0] = (X - (W_rgb/2))/W_rgb
        rgb_xyz[:, :, :, 1] = (Y - (H_rgb/2))/H_rgb
        rgb_xyz[:, :, :, 2] = depth_ims
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
        # if os.path.isfile(op_flow_3D_dir + '/{:05}.npz'.format(vid_id)):
        #     # print("Found 3D optical flow {:05} in cache".format(vid_id))
        #     return np.load(op_flow_3D_dir + '/{:05}.npz'.format(vid_id))['arr_0']

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
            p0v = np.clip(p0v, 0, 479).astype(int)
            p0u = np.clip(p0u, 0, 639).astype(int)

            # Get the points in frame t
            p0 = rgb_xyz[frame, p0v, p0u]

            # Get the displacement vector between p(t) and p(t+1)
            disp_vecs = rgb_xyz[frame+1, p1[0], p1[1]] - p0
            disp_vecs[np.sum(dudv*dudv, axis=1) < 0.5] = np.array([0,0,0])
            disp_vecs[np.abs(disp_vecs[:, 2]) > 0.2] = np.array([0,0,0])
            disp_vecs[(np.abs(disp_vecs[:, 2]) > 0.1) & (np.sum(np.abs(disp_vecs[:, :2]), 1) < 0.01)] = np.array([0,0,0])

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
            np.savez_compressed(op_flow_3D_dir + '/{:05}'.format(vid_id), op_flow_3D_vec)

        return op_flow_3D


    def get_voxel_flow(self, vid_id):
        '''
        Voxelize the 3D optical flow tensor (sparse)
        '''
        VOXEL_SIZE = 54

        # Get 3D optical flow
        op_flow_3D = self.get_3D_optical_flow(vid_id)

        # Pull useful stats out of optical flow
        num_frames = len(op_flow_3D)
        all_xyz = np.vstack(op_flow_3D)
        max_x, max_y, max_z = np.max(all_xyz, axis=0)[:3] + 0.0001
        min_x, min_y, min_z = np.min(all_xyz, axis=0)[:3]

        # Fill in the voxel grid
        voxel_flow_tensor = np.zeros([num_frames, 4, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])
        for frame in range(num_frames):#tqdm(range(num_frames), "Filling in Voxel Grid"):

            # Interpolate and discretize location of the voxels in the grid
            vox_x = np.floor((op_flow_3D[frame][:,0] - min_x)/(max_x - min_x) * VOXEL_SIZE).astype(int)
            vox_y = np.floor((op_flow_3D[frame][:,1] - min_y)/(max_y - min_y) * VOXEL_SIZE).astype(int)
            vox_z = np.floor((op_flow_3D[frame][:,2] - min_z)/(max_z - min_z) * VOXEL_SIZE).astype(int)

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








def create_all_op_flow_3D():
    dataset = SYSU()
    for vid in range(dataset.num_vids):
        x = dataset.get_3D_optical_flow(vid)
        np.savez_compressed(op_flow_3D_dir + '/{:05}'.format(vid), x)


def show_voxel_flow():
    dataset = SYSU()
    vox_flow = dataset.get_voxel_flow(0)
    OpenGlViewer(vox_flow).view()


def main():
    create_all_op_flow_3D()
    # show_voxel_flow()


if __name__ == '__main__':
    main()
