"""
Class to manage building, loading, and saving features for an action
recognition CNN over the NTURGB dataset


Features Implemented
--------------------
- 3D voxel flow
- 3D image as voxel grid
"""


import sys, os
import numpy as np
from ntu_rgb import NTU
from sysu_dataset import SYSU
from tqdm import tqdm, trange
import multiprocessing

from config import *

####################
# Files & Directories
# CACHE_DIR = "/home/mike/Documents/Activity_Recognition/nturgb+d_features_small"
# CACHE_DIR = "/hdd/Datasets/SYSU/voxel_flow_3D_54"
CACHE_DIR = "/hdd/Datasets/NTU/ntu_3D_voxel_images"



####################
# Hyper Parameters
# K -- The number of features per video
K = 5
# T -- The number of frames in each feature
T = 10




class FeatureManager:
    def __init__(self):
        self.dataset = NTU()
        # self.dataset = SYSU()




    def build_feature(self, vid_id):
        """
        Build the specified feature using the dataset wrapper
        """

        # Get the voxel flow from the ntu_rgb wrapper
        vox_flow = self.dataset.get_voxel_flow(vid_id)

        # Split up video into K equal parts of size T
        frames = vox_flow.shape[0]
        skip_amount = (frames - T) / (K - 1)
        features = []
        for feature_idx in range(K):
            start = int(skip_amount * feature_idx)
            end = int(start + T)
            feature = np.vstack(vox_flow[start:end,1:,:,:,:]) # Stack frames
            features.append(feature)

        # Combine all chunks into one tensor
        stacked_feature = np.stack(features)

        return stacked_feature




    def save_feature_sparse(self, feature, vid_id):
        """
        Create the feature, save the non-zero values along with all the data
        needed to fill it back in
        """

        # Get nonzero values
        nonzeros = np.array(np.nonzero(feature))

        # Save the non-zero locations, values, and shape of original feature
        np.save("{}/{:05}.npy".format(CACHE_DIR, vid_id), feature[tuple(nonzeros)])
        np.save("{}/{:05}.nonzeros.npy".format(CACHE_DIR, vid_id), nonzeros)
        np.save("{}/{:05}.shape.npy".format(CACHE_DIR, vid_id), feature.shape)




    def load_feature(self, vid_id):
        """
        Load a feature from the cached data
        """

        # Load the data
        feat_values  = np.load("{}/{:05}.npy".format(CACHE_DIR, vid_id))
        feat_nonzero = np.load("{}/{:05}.nonzeros.npy".format(CACHE_DIR, vid_id))
        feat_shape   = np.load("{}/{:05}.shape.npy".format(CACHE_DIR, vid_id))

        # Rebuild the feature from the saved data
        feature = np.zeros(feat_shape, np.float32)
        feature[tuple(feat_nonzero)] = feat_values

        return feature




    def load_3D_image(self, vid_id):
        VOXEL_SIZE = 108
        feat_nonzero = np.load("{}/{:05}.nonzeros.npy".format(CACHE_3D_IMAGES, vid_id))
        feature = np.zeros([5, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE], np.float32)
        feature[tuple(feat_nonzero)] = 1
        return feature



    def build_and_save_3D_image(self, vid_id):
        VOXEL_SIZE = 108
        op_flow_3D = np.load('{}/{:05}.npz'.format(CACHE_3D_IMAGES, vid_id))['arr_0']
        num_frames = len(op_flow_3D)
        all_xyz = np.vstack(op_flow_3D)
        max_x, max_y, max_z = np.max(all_xyz, axis=0)[:3] + 0.00001
        min_x, min_y, min_z = np.min(all_xyz, axis=0)[:3]
        voxel_flow_tensor = np.zeros([num_frames, 4, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])
        for frame in range(num_frames):
            vox_x = np.floor((op_flow_3D[frame][:,0] - min_x)/(max_x - min_x) * VOXEL_SIZE).astype(np.uint8)
            vox_y = np.floor((op_flow_3D[frame][:,1] - min_y)/(max_y - min_y) * VOXEL_SIZE).astype(np.uint8)
            vox_z = np.floor((op_flow_3D[frame][:,2] - min_z)/(max_z - min_z) * VOXEL_SIZE).astype(np.uint8)
            voxel_flow_tensor[frame, 0, vox_x, vox_y, vox_z] = 1
        skip_amount = (voxel_flow_tensor.shape[0] - T) / (K - 1)
        fancy_idx = [int(skip_amount * i)+5 for i in range(K)]
        feature = voxel_flow_tensor[fancy_idx,0,:,:,:]
        nonzeros = np.array(np.nonzero(feature)).astype(np.uint8)
        np.save("{}/{:05}.nonzeros.npy".format(CACHE_DIR, vid_id), nonzeros)


def main():
    """
    Cache all features
    """
    fm = FeatureManager()
    for x in tqdm(range(fm.dataset.num_vids, desc="Creating features")):
        feature = fm.build_feature(x)
        fm.save_feature_sparse(feature, x)




if __name__ == '__main__':
    main()
