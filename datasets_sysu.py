from sysu_dataset import SYSU

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
K = 5
T = 10

class SYSUdataset(Dataset):
    def __init__(self, test=False, full_train=False):
        # Underlying dataset and features
        self.dataset = SYSU()

        # What to return
        self.images = DATA_IMAGES
        self.images_3D = DATA_IMAGES_3D
        self.op_flow = DATA_OP_FLOW
        self.op_flow_2D = DATA_OP_FLOW_2D
        self.single_feature = DATA_SINGLE_FEAT
        self.augmentation = DATA_AUGMENTATION

        # Train, validation, test split
        self.train = full_train
        if test:
            self.vid_ids = self.dataset.get_splits(SPLIT_NUMBER)[1]
        else:
            self.vid_ids = self.dataset.get_splits(SPLIT_NUMBER)[0]

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

        return torch.from_numpy(op_flow)




    def get_3D_op_flow(self, vid_id):
        # Load the data
        feat_values  = np.load("{}/{:05}.npy".format(CACHE_3D_VOX_FLOW_SYSU, vid_id))
        feat_nonzero = np.load("{}/{:05}.nonzeros.npy".format(CACHE_3D_VOX_FLOW_SYSU, vid_id))
        feat_shape   = np.load("{}/{:05}.shape.npy".format(CACHE_3D_VOX_FLOW_SYSU, vid_id))

        # Rebuild the feature from the saved data
        feature = np.zeros(feat_shape, np.float32)
        feature[tuple(feat_nonzero)] = feat_values

        return feature




    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        to_return = []

        # Images
        if self.images:
            images = np.load('{}/{:05}.npy'.format(CACHE_2D_IMAGES_SYSU, vid_id))
            images = self.image_transforms(images)
            to_return.append(images)

        # Optical flow 3D
        if self.op_flow:
            op_flow = self.get_3D_op_flow(vid_id)
            op_flow = self.op_flow_transforms(op_flow)
            to_return.append(op_flow)

        # Labels
        to_return.append(self.dataset.get_label(vid_id))

        return to_return





def get_train_loader():
    dataset = SYSUdataset(full_train=True)
    return torch.utils.data.DataLoader(dataset, batch_size=DATA_BATCH_SIZE,
                                       shuffle=True, num_workers=NUM_WORKERS,
                                       pin_memory=True)



def get_test_loader():
    dataset = SYSUdataset(test=True)
    return torch.utils.data.DataLoader(dataset, batch_size=DATA_BATCH_SIZE,
                                       shuffle=False, num_workers=NUM_WORKERS,
                                       pin_memory=True)
