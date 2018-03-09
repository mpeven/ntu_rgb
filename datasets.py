from ntu_rgb import NTU
from feature_manager import FeatureManager

import numpy as np
import scipy
import itertools
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import *


vox_size=54
all_tups = np.array(list(itertools.product(range(vox_size), repeat=2)))
rot_array = np.arange(vox_size*vox_size).reshape([vox_size,vox_size])

class NTURGBDataset(Dataset):
    '''
    NTURGB dataset - inherited from pytorch Datset class

    ...

    Attributes
    ----------
    op_flow : bool
        Whether to return the 3D optical flow
    op_flow_2D : bool
        Whether to return the 2D optical flow
    images : bool
        Whether to return the images
    images_3D : bool
        Whether to return the 3D images
    single_feature : bool
        Whether to return a single feature per video or multiple
    test : bool
        Return test set videos
    validation : bool
        Return validation set videos
    full_train : bool
        No validation set - all training data is used
    cross_view : bool
        Whether to return the cross-view split instead of cross-subject
    augmentation : bool
        Whether to perform data augmentation
    '''
    def __init__(self, test=False, validation=False, full_train=False):
        # Underlying dataset and features
        self.dataset = NTU()
        self.features = FeatureManager()

        # What to return
        self.images = DATA_IMAGES
        self.images_3D = DATA_IMAGES_3D
        self.op_flow = DATA_OP_FLOW
        self.op_flow_2D = DATA_OP_FLOW_2D
        self.single_feature = DATA_SINGLE_FEAT
        self.augmentation = DATA_AUGMENTATION

        # Train, validation, test split
        self.train = (test == False) and (validation == False)
        if DATA_CROSS_VIEW == False:
            if test: self.vid_ids = self.dataset.test_split_subject
            elif validation: self.vid_ids = self.dataset.validation_split_subject
            elif full_train: self.vid_ids = self.dataset.train_split_subject
            else: self.vid_ids = self.dataset.train_split_subject_with_validation
        else:
            if test: self.vid_ids = self.dataset.test_split_camera
            else: self.vid_ids = self.dataset.train_split_camera





    def __len__(self):
        return len(self.vid_ids)




    def image_transforms(self, numpy_imgs):
        ''' Transformations on a list of images

        Returns
        -------
        images : Torch Tensor
            Stacked tensor of all images with the transformations applied
        '''

        # Get random parameters to apply same transformation to all images in list
        color_jitter = transforms.ColorJitter.get_params(.25,.25,.25,.25)
        rotation_param = transforms.RandomRotation.get_params((-15,15))
        crop_params = None

        # Apply transformations
        images = []
        for numpy_img in numpy_imgs:
            i = transforms.functional.to_pil_image(numpy_img)
            i = transforms.functional.resize(i, (224,224))
            if self.train:
                i = color_jitter(i)
                i = transforms.functional.rotate(i, rotation_param)
            i = transforms.functional.to_tensor(i)
            i = transforms.functional.normalize(i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            images.append(i)
        return torch.stack(images)




    def op_flow_transforms(self, op_flow):
        ''' Transformations on a tensor of optical flow voxel grids

        Parameters
        ----------
        op_flow : ndarray

        Returns
        -------
        op_flow : Torch Tensor
            A torch tensor of an optical flow voxel grid with the
            transformations (rotation, scale, translation) applied to it
        '''
        def translate(op_flow):
            # op_flow[:,0::3,:,:,:] ---> x axis vectors
            # op_flow = scipy.ndimage.interpolation.shift(op_flow, [0,0,x_move,y_move,z_move], cval=0, order=0) # Slower alternative
            # Get amount to shift
            max_shift = int(op_flow.shape[2] * 0.10)
            x_move, y_move, z_move = np.random.randint(-max_shift, max_shift, 3)

            # Translate values
            if x_move > 0:
                op_flow[:,:,x_move:,:,:] = op_flow[:,:,:-x_move,:,:]
                op_flow[:,:,:x_move,:,:] = 0
            elif x_move < 0:
                op_flow[:,:,:x_move,:,:] = op_flow[:,:,-x_move:,:,:]
                op_flow[:,:,x_move:,:,:] = 0
            if y_move > 0:
                op_flow[:,:,:,y_move:,:] = op_flow[:,:,:,:-y_move,:]
                op_flow[:,:,:,:y_move,:] = 0
            elif y_move < 0:
                op_flow[:,:,:,:y_move,:] = op_flow[:,:,:,-y_move:,:]
                op_flow[:,:,:,y_move:,:] = 0
            if z_move > 0:
                op_flow[:,:,:,:,z_move:] = op_flow[:,:,:,:,:-z_move]
                op_flow[:,:,:,:,:z_move] = 0
            elif z_move < 0:
                op_flow[:,:,:,:,:z_move] = op_flow[:,:,:,:,-z_move:]
                op_flow[:,:,:,:,z_move:] = 0
            return op_flow


        def rotate(op_flow):
            ''' Rotate an optical flow tensor a random amount about the y axis '''
            # Get angle
            angle = np.random.randint(-45, 45)

            # Rotate positions
            rot_mat = scipy.ndimage.interpolation.rotate(rot_array, angle, (0,1), reshape=False, order=0)
            op_flow_new = np.zeros(op_flow.shape, dtype=np.float32)
            tup = all_tups[rot_mat]
            op_flow_new = op_flow[:,:,tup[:, :, 0],:,tup[:, :, 1]].transpose(2,3,0,4,1)

            # Rotate flow vectors
            cos = np.cos(np.radians(-angle))
            sin = np.sin(np.radians(-angle))
            x_copy = op_flow_new[:,0].copy()
            z_copy = op_flow_new[:,2].copy()
            op_flow_new[:,0] = x_copy * cos + z_copy * sin
            op_flow_new[:,2] = x_copy * -sin + z_copy * cos

            return op_flow_new


        def scale(op_flow):
            return op_flow

        # import datetime as dt
        if self.train:
            op_flow = translate(op_flow)
            op_flow = rotate(op_flow)
            # op_flow = scale(op_flow)

        return torch.from_numpy(op_flow)




    def __getitem__(self, idx):
        to_return = []
        vid_id = self.vid_ids[idx]

        # Images
        if self.images:
            images = np.load('{}/{:05}.npy'.format(CACHE_2D_IMAGES, vid_id))
            if self.augmentation:
                images = self.image_transforms(images)
            if self.single_feature:
                images = images[2]
            to_return.append(images)

        # 3D images
        if self.images_3D:
            images_3D = np.load('{}/{:05}.npy'.format(CACHE_3D_IMAGES, vid_id))
            if self.single_feature:
                images_3D = images_3D[2]

        # Optical flow 3D
        if self.op_flow:
            op_flow = self.features.load_feature(vid_id)
            if self.augmentation:
                op_flow = self.op_flow_transforms(op_flow)
            if self.single_feature:
                op_flow = op_flow[2]
            to_return.append(op_flow)

        # Optical flow 2D
        if self.op_flow_2D:
            # op_flow_2D = np.zeros([50, 400, 400, 2])
            # for i in range(50):
            #     im = cv2.imread('{}/{:05}/{:02}.png'.format(CACHE_2D_OP_FLOW_PNG, vid_id, i))
            #     op_flow_2D[i,:,:,0:1] = im[:,:,0:1]
            #
            # # Rescale & Reshape
            # m0, m1 = np.load('{}/{:05}/min_max.npy'.format(CACHE_2D_OP_FLOW_PNG, vid_id))
            # op_flow_2D = (((op_flow_2D/255.)*(m1-m0))-np.abs(m0)).reshape([5,20,400,400]).astype(np.float32)
            op_flow_2D = np.load('/hdd/Datasets/NTU/nturgb+d_op_flow_2D_small/{:05}.npy'.format(vid_id))
            if self.single_feature:
                op_flow_2D = op_flow_2D[2]
            to_return.append(op_flow_2D)

        # Label
        y = self.dataset.id_to_action[vid_id]
        to_return.append(y)

        return to_return





def get_train_valid_loader():
    # Create the dataset
    train_dataset = NTURGBDataset()
    valid_dataset = NTURGBDataset(validation=True)

    # Seed the shuffler
    np.random.seed(149)
    torch.manual_seed(149)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=DATA_BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=DATA_BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True)

    return (train_loader, valid_loader)





def get_train_loader():
    dataset = NTURGBDataset(full_train=True)
    return torch.utils.data.DataLoader(dataset, batch_size=DATA_BATCH_SIZE,
                                       shuffle=True, num_workers=NUM_WORKERS,
                                       pin_memory=True)



def get_test_loader():
    dataset = NTURGBDataset(test=True)
    return torch.utils.data.DataLoader(dataset, batch_size=DATA_BATCH_SIZE,
                                       shuffle=False, num_workers=NUM_WORKERS,
                                       pin_memory=True)
