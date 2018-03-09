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
from tqdm import tqdm


####################
# Files & Directories
CACHE_DIR = "/home/mike/Documents/Activity_Recognition/nturgb+d_features_small"



####################
# Hyper Parameters
# K -- The number of features per video
K = 5
# T -- The number of frames in each feature
T = 10




class FeatureManager:
    def __init__(self):
        self.dataset = NTU()




    def build_feature(self, vid_id, voxel_flow=True, image_3D=False):
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

            ### Voxel flow
            if voxel_flow:
                feature = np.vstack(vox_flow[start:end,1:,:,:,:]) # Stack frames

            ### 3D voxel image
            if image_3D:
                feature = vox_flow[start+(10//2),0,:,:,:]

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




def main():
    """
    Cache all features
    """
    fm = FeatureManager()
    for x in tqdm(range(fm.dataset.num_vids), "Creating features"):
        feature = fm.build_feature(x)
        fm.save_feature(feature, x)




if __name__ == '__main__':
    main()
