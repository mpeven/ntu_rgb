#!/usr/bin/env python

import os, sys
import numpy as np
import cv, cv2
import h5py
import math

from os import listdir
from os.path import isfile, join

sys.path.append('..')

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_conv3d, dnn_pool

from lib import inits
from lib import updates
from lib import activations
from lib.rng import np_rng
from lib.ops import batchnorm, deconv, batchnorm3d
from lib.theano_utils import floatX, sharedX

from sklearn.externals import joblib
from time import time
import json

from utils import save_ply





##############################################################################################################
# Some pre-defined Kinect v1 parameters

# The maximum depth used, in meters.
maxDepth = 10

#  RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02
cy_rgb = 2.5373616633400465e+02

# RGB Distortion Parameters
k1_rgb =  2.0796615318809061e-01
k2_rgb = -5.8613825163911781e-01
p1_rgb = 7.2231363135888329e-04
p2_rgb = 1.0479627195765181e-03
k3_rgb = 4.9856986684705107e-01

# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02
fy_d = 5.8269103270988637e+02
cx_d = 3.1304475870804731e+02
cy_d = 2.3844389626620386e+02

# Depth Distortion Parameters
k1_d = -9.9897236553084481e-02
k2_d = 3.9065324602765344e-01
p1_d = 1.9290592870229277e-03
p2_d = -1.9422022475975055e-03
k3_d = -5.1031725053400578e-01

# Rotation
R = -np.array([[-1.0000, 0.0050, 0.0043],
    [-0.0051, -1.0000, -0.0037],
    [-0.0043, 0.0037, -1.0000]])

# 3D Translation (meters)
t_x = 2.5031875059141302e-02
t_z = -2.9342312935846411e-04
t_y = 6.6238747008330102e-04

# Parameters for making depth absolute
depthParam1 = 351.3
depthParam2 = 1092.5
#
##############################################################################################################





##############################################################################################################
def deg2rad(degs):
    """
    Convert an angle from degrees to radians.
    """
    return degs*np.pi/180.0


##############################################################################################################
def rad2deg(rads):
    """
    Convert an angle from radians to degrees.
    """
    return rads*180.0/np.pi



##############################################################################################################
def rot_angles_to_mat(xr, yr, zr):
    """
    Convert a set of rotation angles in x, y, and z into a 3x3 orthonormal rotation matrix.
    We assume the input angles are in radians.
    """

    Rx = np.eye(3)
    Ry = np.eye(3)
    Rz = np.eye(3)

    Rx[1,1] = np.cos(xr)
    Rx[1,2] = -np.sin(xr)
    Rx[2,1] = np.sin(xr)
    Rx[2,2] = np.cos(xr)

    Ry[0,0] = np.cos(yr)
    Ry[0,2] = np.sin(yr)
    Ry[2,0] = -np.sin(yr)
    Ry[2,2] = np.cos(yr)

    Rz[0,0] = np.cos(zr)
    Rz[0,1] = -np.sin(zr)
    Rz[1,0] = np.sin(zr)
    Rz[1,1] = np.cos(zr)

    return np.dot(Rz,np.dot(Ry,Rx))




##############################################################################################################
def transform_tsdf(src, minx, miny, minz, voxel_size, T):

    ix, iy, iz = np.where(src[:,:,:,0] < 1.0)

    Nx = src.shape[0]
    Ny = src.shape[1]
    Nz = src.shape[2]

    px = minx + ix*voxel_size
    py = miny + iy*voxel_size
    pz = minz + iz*voxel_size

    pts = np.vstack((px,py,pz)).T  # Nx3
    t_pts = np.dot(pts, T[:3,:3]) + np.repeat(np.expand_dims(T[:3,3],axis=0), pts.shape[0], axis=0)  # Nx3

    dst = np.ones(src.shape, src.dtype)

    jx = np.floor((t_pts[:,0]-minx)/voxel_size).astype(np.int32)
    jy = np.floor((t_pts[:,1]-miny)/voxel_size).astype(np.int32)
    jz = np.floor((t_pts[:,2]-minz)/voxel_size).astype(np.int32)

    valid = np.where(np.bitwise_and(jx>=0, np.bitwise_and(jy>=0, np.bitwise_and(jz>=0,
        np.bitwise_and(jx<Nx, np.bitwise_and(jy<Ny, jz<Nz))))))[0]

    jx = jx[valid]
    jy = jy[valid]
    jz = jz[valid]

    ix = ix[valid]
    iy = iy[valid]
    iz = iz[valid]

    dst[jx[:], jy[:], jz[:]] = src[ix[:], iy[:], iz[:]].copy()
    return dst


##############################################################################################################
def tsdf(depth_im, Nx, Ny, Nz, minx, miny, minz, voxel_size):

    VG = np.ones((Nx,Ny,Nz,1), dtype=np.float32)

    im_width = depth_im.shape[1]
    im_height = depth_im.shape[0]

    trunc_margin = voxel_size * 5.0

    maxx = minx + Nx*voxel_size
    maxy = miny + Ny*voxel_size
    maxz = minz + Nz*voxel_size

    xx, yy, zz = np.meshgrid(np.linspace(minx,maxx,Nx), np.linspace(miny,maxy,Ny), np.linspace(minz,maxz,Nz))
    ix, iy, iz = np.meshgrid(range(Nx), range(Ny), range(Nz))

    xx = xx.flatten()
    yy = yy.flatten()
    zz = zz.flatten()

    ix = ix.flatten()
    iy = iy.flatten()
    iz = iz.flatten()

    xn = xx/zz
    yn = yy/zz

    u = np.round(fx_d*xn + cx_d).astype(np.int32)
    v = np.round(fy_d*yn + cy_d).astype(np.int32)

    valid = np.where(np.bitwise_and(u>=0, np.bitwise_and(v>=0, np.bitwise_and(u<im_width, v<im_height))))[0]
    gridx = xx[valid]
    gridy = yy[valid]
    gridz = zz[valid]
    #
    ix = ix[valid]
    iy = iy[valid]
    iz = iz[valid]
    #
    u = u[valid]
    v = v[valid]
    #
    xn = xn[valid]
    yn = yn[valid]

    depthz = depth_im[v[:],u[:]].flatten()

    valid = np.where(np.bitwise_and(depthz>0, depthz<=6))[0]
    gridx = gridx[valid]
    gridy = gridy[valid]
    gridz = gridz[valid]
    #
    ix = ix[valid]
    iy = iy[valid]
    iz = iz[valid]
    #
    xn = xn[valid]
    yn = yn[valid]
    #
    depthz = depthz[valid]

    diff = (depthz-gridz) * np.sqrt(1.0 + np.power(xn,2) + np.power(yn,2))

    valid = np.where(diff>-trunc_margin)[0]
    ix = ix[valid]
    iy = iy[valid]
    iz = iz[valid]
    #
    diff = diff[valid]

    VG[ix[:],iy[:],iz[:],0] = np.minimum(np.ones(diff.shape), diff/trunc_margin)
    return VG



# Set voxel dimensions for constructing the TSDF grid
#Nx = 208
#Ny = 100
#Nz = 208
#
Nx = 104
Ny = 50
Nz = 104


# Load in the image files and associated labels for training
train_label_dict = {}
train_files = []
train_labels = []
lines = open('nyurgbd_train.txt', 'r').readlines()
for l in lines:
    if len(l) == 0:
        continue
    S = l.split()

    train_files.append(S[0])

    lab = int(S[1])
    train_labels.append(lab)

    if not train_label_dict.has_key(lab):
        train_label_dict[lab] = lab
nImages = len(train_files)
nb_classes = len(train_label_dict.keys())


relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
softmax = activations.Softmax()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

desc = 'ntu_3d_cnn_classify'
model_dir = 'models/%s'%desc
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Initialize parameters for base CNN model
dw  = difn((96, 1, 3, 3, 3), 'dw')       # [nfilters x n_incoming_channels x filter_x x filter_y x filter_z]: In=[1x104x50x104], Out=[96x52x25x52]
dg = gain_ifn((96), 'dg')
db = bias_ifn((96), 'db')

dw2 = difn((196, 96, 3, 3, 3), 'dw2')    # In=[96x52x25x52], Out=[196x26x12x26]
dg2 = gain_ifn((196), 'dg2')
db2 = bias_ifn((196), 'db2')

dw3 = difn((384, 196, 3, 3, 3), 'dw3')   # In=[196x26x12x26], Out=[384x13x6x13]
dg3 = gain_ifn((384), 'dg3')
db3 = bias_ifn((384), 'db3')

dw4 = difn((512, 384, 3, 3, 3), 'dw4')   # In=[384x13x6x13], Out=[512x6x3x6]
dg4 = gain_ifn((512), 'dg4')
db4 = bias_ifn((512), 'db4')

dfc1 = difn((512*6*3*6, 2048), 'dfc1')   # In=[512*6*3*6,], Out=[2048,]
dfc2 = difn((2048, 2048), 'dfc2')        # In=[2048,], Out=[2048]
dy = difn((2048, nb_classes), 'dy')      # In=[2048,], Out=[nb_classes,]

model_params = [dw, dg, db, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dfc1, dfc2, dy]
start_epoch = 0


#model_params = [sharedX(p) for p in joblib.load(model_dir+'/8_model_params.jl')]
#start_epoch = 9


def classify(X, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wfc1, wfc2, wy):

    h = relu(batchnorm3d(dnn_conv3d(X, w, subsample=(1, 1, 1), border_mode=(1, 1, 1)), g=g, b=b))
    h = dnn_pool(h, ws=(2, 2, 2), stride=(2, 2, 2), mode='max')

    h2 = relu(batchnorm3d(dnn_conv3d(h, w2, subsample=(1, 1, 1), border_mode=(1, 1, 1)), g=g2, b=b2))
    h2 = dnn_pool(h2, ws=(2, 2, 2), stride=(2, 2, 2), mode='max')

    h3 = relu(batchnorm3d(dnn_conv3d(h2, w3, subsample=(1, 1, 1), border_mode=(1, 1, 1)), g=g3, b=b3))
    h3 = dnn_pool(h3, ws=(2, 2, 2), stride=(2, 2, 2), mode='max')

    h4 = relu(batchnorm3d(dnn_conv3d(h3, w4, subsample=(1, 1, 1), border_mode=(1, 1, 1)), g=g4, b=b4))
    h4 = dnn_pool(h4, ws=(2, 2, 2), stride=(2, 2, 2), mode='max')

    h5 = T.flatten(h4, 2)

    h6 = relu(T.dot(h5, wfc1))
    h7 = relu(T.dot(h6, wfc2))

    y = softmax(T.dot(h7, wy))
    return y



Xtsdf = T.tensor5()  # input voxel grid
ylabels = T.matrix() # ground truth classification labels

predictions = classify(Xtsdf,
    model_params[0], model_params[1], model_params[2], model_params[3], model_params[4], model_params[5],
    model_params[6], model_params[7], model_params[8], model_params[9], model_params[10], model_params[11],
    model_params[12], model_params[13], model_params[14])

cost_y = cce(predictions, ylabels).mean()
cost = [cost_y]

updater = updates.Adam()
updates = updater(model_params, cost_y)

print 'COMPILING'
t = time()
_train_ = theano.function([Xtsdf, ylabels], cost, updates=updates)
print '%.2f seconds to compile theano functions\n'%(time()-t)



print '\nNum training images:', nImages, '--- Number of class labels:', nb_classes, '\n'

nb_epochs = 50
nb_samples_per_epoch = 75000
nbatch = 6

Xmb = np.zeros((nbatch,1,Nx,Ny,Nz), dtype=np.float32)
ymb = np.zeros((nbatch,nb_classes), dtype=np.float32)
Htransform = np.eye(4, dtype=np.float32)


# Set global bounds on the voxel grid to provide consistency across training samples
gminx = -2.6
gminy = -1.5
gminz = 0.4
#voxel_size = 0.025
voxel_size = 0.05



for epoch in range(start_epoch,nb_epochs):

    sumCost = 0.0
    Nd = 0.0

    for i in range(0,nb_samples_per_epoch,nbatch):
        ilow = i
        ihigh = min(ilow+nbatch, nb_samples_per_epoch)
        Nb = ihigh-ilow
        rand_inds = np.random.randint(0, high=nImages, size=Nb)

        Xmb[:] = 0
        ymb[:] = 0

        for b in range(Nb):
            j = rand_inds[b]
            this_label = train_labels[j]
            depth_img = cv2.imread(train_files[j], -1)        # <--- read it in "as is",data format-wise

            # Sometimes the image doesn't exist or is corrupt.  If that happens, re-randomize and try again
            while depth_img is None:
                rand_inds = np.random.randint(0, high=nImages, size=Nb)

                j = rand_inds[b]
                this_label = train_labels[j]
                depth_img = cv2.imread(train_files[j], -1)

            depth_img.byteswap(True)                          # need to do this to swap big <--> little endian
            depth_img = depth_img.astype(np.float32)
            depth_img = 351.3 / (1092.5 - depth_img)          # convert from raw Kinect values to meters

            # Add some Gaussian noise to the depth values to make a little robust to sensor noise
            depth_img += np.random.randn(depth_img.shape[0],depth_img.shape[1])*0.05

            # Get the TSDF for this depth image
            this_tsdf = tsdf(depth_img, Nx, Ny, Nz, gminx, gminy, gminz, voxel_size)

            # With some chance, apply a random rigid transformation to the voxel grid for
            # real-time data augmentation
            if np.random.rand() <= 0.4:
                rx = float(np.random.randint(-20, high=21))           # x-rotation, in degs  --> [-20,20] degs
                ry = float(np.random.randint(-20, high=21))           # y-rotation, in degs  --> [-20,20] degs
                rz = float(np.random.randint(-20, high=21))           # z-rotation, in degs  --> [-20,20] degs
                tx = float(np.random.randint(-40, high=41))/100.0     # x-translation, in m  --> [-0.4,0.4] meters
                ty = float(np.random.randint(-40, high=41))/100.0     # y-translation, in m  --> [-0.4,0.4] meters
                tz = float(np.random.randint(-40, high=41))/100.0     # z-translation, in m  --> [-0.4,0.4] meters

                Htransform[:3,:3] = rot_angles_to_mat(deg2rad(rx), deg2rad(ry), deg2rad(rz))
                Htransform[0,3] = tx
                Htransform[1,3] = ty
                Htransform[2,3] = tz

                dst_tsdf = transform_tsdf(this_tsdf, gminx, gminy, gminz, voxel_size, Htransform)
                this_tsdf = dst_tsdf.copy()

            # out_pc = []
            # min_val = this_tsdf.min()
            # max_val = this_tsdf.max()
            # for xx in range(Nx):
            #   for yy in range(Ny):
            #       for zz in range(Nz):
            #           if this_tsdf[xx,yy,zz] < 1.0:
            #               val = int(255.0*(this_tsdf[xx,yy,zz]-min_val)/(max_val-min_val))
            #               out_pc.append([xx,yy,zz,val,val,val])
            # save_ply(np.array(out_pc), 'tsdf.ply')
            # print train_files[j]
            # print 8/0

            # Store the voxel grid into the batch (down-sampled)
            #Xmb[b] = this_tsdf[0::2,0::2,0::2,:].transpose((3,0,1,2))
            Xmb[b] = this_tsdf.transpose((3,0,1,2))

            # And the associated label
            ymb[b,this_label] = 1.0

        # Train
        cost = _train_(Xmb, ymb)

        sumCost += float(cost[0])
        Nd += 1.0

        print "\rEpoch: {:02d} | Mean-Cost {:2.5f} | Samples {:07d}/{:07d}".format(
                epoch, sumCost/Nd, i+1, nb_samples_per_epoch),
        sys.stdout.flush()

    print '\n'
    joblib.dump([p.get_value() for p in model_params], 'models/%s/%d_model_params.jl'%(desc, epoch))

