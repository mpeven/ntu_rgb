from ntu_rgb import NTU
import numpy as np
import os
from tqdm import tqdm
import sys
from skimage.transform import resize


def save_ims(start_idx, end_idx):
    dataset = NTU()

    for vid in tqdm(range(start_idx, end_idx)):
        if vid >= dataset.num_vids:
            exit()

        # Get ndarray of images
        all_images = dataset.get_rgb_vid_images(vid)
        skip_amount = (all_images.shape[0] - 10) / (5 - 1)
        images = []
        for feature_idx in range(5):
            start_im = int(skip_amount * feature_idx)
            image = all_images[start_im+(10//2)]
            images.append(image)
        full_ims = np.stack(images)

        # Crop the images
        bbox = dataset.get_rgb_mask(vid)
        cropped_ims = full_ims[:,bbox[0]:bbox[2],bbox[1]:bbox[3]]
        np.save('/home-3/mpeven1@jhu.edu/data/nturgb+d_rgb_masked/{:05}'.format(vid), cropped_ims)




def save_optical_flow(start_idx, end_idx):
    dataset = NTU()

    for vid in tqdm(range(start_idx, end_idx)):
        if vid >= dataset.num_vids:
            exit()

        if os.path.isfile('/home-3/mpeven1@jhu.edu/work/dev_mp/nturgb_cache/optical_flow_2D/{:05}.npz'.format(vid)):
            continue

        # Get optical flow
        full_flow = dataset.get_2D_optical_flow(vid)

        # Mask the optical flow
        bbox = dataset.get_rgb_mask(vid)
        cropped_op_flow = full_flow[:,:,bbox[0]:bbox[2],bbox[1]:bbox[3]]

        # Filter the optical flow
        cropped_op_flow[np.stack([np.linalg.norm(cropped_op_flow, axis=1)] * 2, axis=1) < 0.5] = 0

        # Resize the optical flow and images
        resized_op_flow1 = resize(cropped_op_flow[:, 0], [cropped_op_flow.shape[0], 400, 400], preserve_range=True)
        resized_op_flow2 = resize(cropped_op_flow[:, 1], [cropped_op_flow.shape[0], 400, 400], preserve_range=True)
        resized_op_flow = np.stack([resized_op_flow1, resized_op_flow2], axis=1)
        op_flow_2D = resized_op_flow.copy().astype(np.float32)

        # Pull out only the frames we care about
        skip_amount = (op_flow_2D.shape[0] - 10) / (5 - 1)
        op_flows = []
        for feature_idx in range(5):
            start = int(skip_amount * feature_idx)
            end = int(start + 10)
            feature = np.vstack(op_flow_2D[start:end]) # Stack frames
            op_flows.append(feature)
        op_flow_2D = np.stack(op_flows)

        np.savez_compressed('/home-3/mpeven1@jhu.edu/work/dev_mp/nturgb_cache/optical_flow_2D/{:05}'.format(vid), op_flow_2D)




def save_3D_optical_flow(start_idx, end_idx):
    dataset = NTU()

    for vid in range(start_idx, end_idx):
        if vid >= dataset.num_vids: exit()
        if os.path.isfile('/home-3/mpeven1@jhu.edu/work/dev_mp/nturgb_cache/optical_flow_3D/{:05}.npz'.format(vid)):
            continue
        op_flow_3D = dataset.get_3D_optical_flow(vid)
        np.savez_compressed('/home-3/mpeven1@jhu.edu/work/dev_mp/nturgb_cache/optical_flow_3D/{:05}'.format(vid), op_flow_3D)



def main():
    # Run with --array=0-5680:10
    start_idx = int(sys.argv[1])*10
    end_idx   = int(sys.argv[1])*10 + 100
    save_3D_optical_flow(start_idx, end_idx)


if __name__ == '__main__':
    main()
