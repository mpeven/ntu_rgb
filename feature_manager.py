"""
Class to manage building, loading, and saving features for an action
recognition CNN over the NTURGB dataset


Features
--------
The features are subsections of each video represented as voxel flow
between each frame in the subsection
"""


import sys, os
import numpy as np
from ntu_rgb import NTU



####################
# Files & Directories

CACHE_DIR = "/hdd/Datasets/NTU/nturgb+d_features"

#
####################



####################
# Hyper Parameters

# K -- The number of features per video
K = 5

# T -- The number of frames in each feature
T = 10

#
####################




class FeatureManager:
    def __init__(self):
        self.dataset = NTU()




    def build_feature(self, vid_id):
        """
        Get the voxel flow, split it up into chunks, stack the chunks
        """

        # Get the voxel flow from the ntu_rgb wrapper
        vox_flow = dataset.get_voxel_flow(vid_id)

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




    def save_feature(self, vid_id):
        """
        Create the feature, save the non-zero values along with all the data
        needed to fill it back in
        """

        # Build the feature
        feature = self.build_feature(vid_id)

        # Get nonzero values
        nonzeros = np.array(np.nonzero(feature))

        # Save the non-zero locations, values, and shape of original feature
        np.save("{}{:05}.npy".format(CACHE_DIR, vid_id), feature[tuple(nonzeros)])
        np.save("{}{:05}.nonzeros.npy".format(CACHE_DIR, vid_id), nonzeros)
        np.save("{}{:05}.shape.npy".format(CACHE_DIR, vid_id), feature.shape)




    def load_feature(self, vid_id):
        """
        Load a feature from the cached data
        """

        # Load the data
        feat_values = np.load("{}{:05}.npy".format(CACHE_DIR, vid_id))
        feat_nonzero = np.load("{}{:05}.nonzeros.npy".format(CACHE_DIR, vid_id))
        feat_shape = np.load("{}{:05}.shape.npy".format(CACHE_DIR, vid_id))

        # Rebuild the feature from the saved data
        feature = np.zeros(feat_shape)
        feature[tuple(feat_nonzero)] = feat_values

        return feature




def main():
    """
    Cache all features
    """
    fm = FeatureManager()
    for x in range(10000):
        fm.save_feature(x)




if __name__ == '__main__':
    main()
