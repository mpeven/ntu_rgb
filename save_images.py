from ntu_rgb import NTU
import numpy as np
import os
from tqdm import tqdm

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


def main():
    # Run with --array=0-56800:100
    start_idx = int(sys.argv[1])
    end_idx   = int(sys.argv[1]) + 100


if __name__ == '__main__':
    main()
