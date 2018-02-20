from ntu_rgb import NTU
from feature_manager import FeatureManager

import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class NTURGBDataset(Dataset):
    '''
    NTURGB dataset - inherited from pytorch Datset class

    ...

    Attributes
    ----------
    single_feature : bool
        Whether to return a single feature per video or multiple
    optical_flow : bool
        Whether to return the optical flow along with the images
    test : bool
        Return test set videos
    validation : bool
        Return validation set videos
    full_train : bool
        No validation set - all training data is used
    '''
    def __init__(self, single_feature=False, optical_flow=False, test=False,
                 validation=False, full_train=False):
        # Underlying dataset and features
        self.dataset = NTU()
        self.features = FeatureManager()

        # What to return
        self.single_feature = single_feature
        self.optical_flow = optical_flow

        # Train, validation, test split
        if test: self.vid_ids = self.dataset.test_split_subject
        elif validation: self.vid_ids = self.dataset.validation_split_subject
        elif full_train: self.vid_ids = self.dataset.train_split_subject
        else: self.vid_ids = self.dataset.train_split_subject_with_validation

        # Transformation from ndarray to image
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.ColorJitter(.1,.1,.1,.1),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]

        # Images
        images = np.load('./nturgb+d_rgb_masked/{:05}.npy'.format(vid_id))
        images = torch.stack([self.transform(i) for i in images])

        # # Transformations
        # x = transforms.ColorJitter(.1,.1,.1,.1)
        # y = transforms.ToPILImage()
        # print(x)
        # print(x.__class__)
        # # print(x.get_params())
        # print(x(y(images[0])))
        # exit()

        # Optical flow
        if self.optical_flow:
            op_flow = torch.from_numpy(self.features.load_feature(vid_id).astype(np.float32))

        # Label
        y = self.dataset.id_to_action[vid_id]

        # Single vs. multiple features
        if self.single_feature:
            images = images[2]
            op_flow = op_flow[2]

        if self.optical_flow:
            return images, op_flow, y
        else:
            return images, y
