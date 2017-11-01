import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import provider
import pc_util

sys.path.append('../dcgan_code')
from lib import caffe_pb2
import lmdb
from utils import collect_files


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
FLAGS = parser.parse_args()


BATCH_SIZE = 64  #4
NUM_POINT = 2048  # number of 3D points per point cloud -- should be consistent with how things were trained
MODEL_PATH = FLAGS.model_path
GPU_INDEX = 0
MODEL = importlib.import_module(FLAGS.model) # import network module


##############################################################################################################
# Some pre-defined Kinect v2 parameters

# The maximum depth used, in meters.
maxDepth = 10

#  RGB Intrinsic Parameters
fx_rgb = 1.0820918205955327e+03
fy_rgb = 1.0823759977994873e+03
cx_rgb = 9.6692449993169430e+02
cy_rgb = 5.3566594698237566e+02

# RGB Distortion Parameters
#
# [ignoring for now]
#

# Depth Intrinsic Parameters
fx_d = 365.481
fy_d = 365.481
cx_d = 257.346
cy_d = 210.347

# Depth Distortion Parameters
#
# [ignoring for now]
#

# Rotation [depth-to-color] ; should this be transposed?
R = -np.array([[9.9997086703565385e-01, -5.6336988734456616e-03, -5.1503899819367671e-03],
               [5.6034256549301270e-03,  9.9996705123920560e-01, -5.8735046520488696e-03],
               [5.1833098395106967e-03,  5.8444737120895603e-03,  9.9996948724755408e-01]])

# 3D Translation (meters)
t_x = 5.2236787502888557e-02
t_y = -3.0543659328634320e-04
t_z = -9.6233443531043941e-04
#
##############################################################################################################



##############################################################################################################
def depth_to_pc(depth_file, N=1024):
    """
    Convert a Kinect depth image to a point cloud.  Input is a depth image file.
    Output is an [N x 3] matrix of 3D points in units of meters.

    (The point cloud is normalized to zero mean and to fit within the unit circle)
    """

    imgDepthAbs = cv2.imread(depth_file, -1)             # read "as is", in mm
    imgDepthAbs = imgDepthAbs.astype(np.float32)/1000.0  # convert to meters

    H = imgDepthAbs.shape[0]
    W = imgDepthAbs.shape[1]

    xx, yy, = np.meshgrid(range(0,W), range(0,H))

    # Just collect valid points
    valid = imgDepthAbs > 0
    xx = xx[valid]
    yy = yy[valid]
    imgDepthAbs = imgDepthAbs[valid]

    npoints = len(imgDepthAbs)

    # Project all depth pixels into 3D (in the depth camera's coordinate system)
    points3d = np.zeros((N,3), dtype=np.float32)
    X = (xx - cx_d) * imgDepthAbs / fx_d
    Y = (yy - cy_d) * imgDepthAbs / fy_d
    Z = imgDepthAbs

    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # If we have more than enough points, randomly sample points here.  Otherwise leave as is
    # with zero-padding.
    if npoints > N:
        rand_perm = np.random.permutation(npoints)
        points3d[:,0] = X[rand_perm[:N]]
        points3d[:,1] = Y[rand_perm[:N]]
        points3d[:,2] = Z[rand_perm[:N]]
    elif npoints <= N:
        points3d[:npoints,0] = X.copy()
        points3d[:npoints,1] = Y.copy()
        points3d[:npoints,2] = Z.copy()

    # Zero mean the points
    m = np.mean(points3d, axis=0)
    points3d[:,0] -= m[0]
    points3d[:,1] -= m[1]
    points3d[:,2] -= m[2]

    # Now get the point with the maximum length and normalize each point to that length
    # so all points fit within the unit sphere.
    max_vec_len = np.linalg.norm(points3d, axis=1).max()
    points3d /= max_vec_len

    return points3d



maxlen = 300     # maximum number of frames in a sequence
#maxlen = 100     # maximum number of frames in a sequence
#dt = 3           # the time step (in number of frames)

nfeat = 1024     # dimensions of feature from PointNet
#nfeat = 256     # dimensions of feature from PointNet



# Where are all of the RGB videos stored?  Retrieve all filenames:
rgb_dir = '/media/austin/Drive/NTU-RGBD-Action-Recognition/rgb/nturgb+d_rgb'

# Where are the depth sequences?  Use the masked depth images to improve
# resolution at the location of the human performing the activity.
depth_dir = '/media/austin/Drive/NTU-RGBD-Action-Recognition/masked_depth/nturgb+d_depth_masked'
#depth_dir = '/media/austin/Drive/NTU-RGBD-Action-Recognition/depth/nturgb+d_depth'

# We need to parse the filenames to figure out the training label as well as
# to see if it's a training sample or testing sample.  As per the standard
# split in the paper, these are the training IDs:
train_IDs = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]




##############################################################################################################
def go():
    is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        _, _, feat = MODEL.get_model_with_feature(pointclouds_pl, is_training_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)

    ops = {'pointclouds_pl': pointclouds_pl,
           'is_training_pl': is_training_pl,
           'feat': feat}

    is_training = False

    video_files = collect_files(rgb_dir, file_ext='.avi')
    nVideos = len(video_files)
    print '\nFound', nVideos, 'video files'

    # Let's load in the video filenames for training.
    print 'Collecting training samples...'
    train_video_files = []
    actions = {}
    for fn in video_files:
        # Get the prefix from the filename so we can parse it
        bn = os.path.basename(fn)
        prefix = os.path.splitext(bn)[0]

        # Each filename is of the form: SsssCcccPpppRrrrAaaa where:
        # sss is the setup number
        # ccc is the camera ID
        # ppp is the performer ID
        # rrr is the replication number
        # aaa is the action class label
        setup_number = int(prefix[1:4])
        camera_id = int(prefix[5:8])
        performer_id = int(prefix[9:12])
        replication_number = int(prefix[13:16])
        action_label = int(prefix[17:20])

        # Also collect the unique action labels
        if not actions.has_key(action_label):
            actions[action_label] = action_label

        # Is this in the training set?
        if performer_id in train_IDs:
            train_video_files.append(fn)
    nTrain = len(train_video_files)
    print '...found', nTrain, 'training samples...'
    print '...done.\n'

    # Setup LMDB for features
    lmdb_loc = '/media/austin/Drive/ntu_feat3d_subject/ntu_feat3d_train_lmdb'
    nbytes = nTrain*maxlen*nfeat*4
    map_size = nbytes * 4
    env = lmdb.open(lmdb_loc, map_size=map_size)

    # Also create a log file to keep track of what file is what
    ofs = open('/media/austin/Drive/ntu_feat3d_subject/ntu_feat3d_train_log.txt', 'w')

    Xmf = np.zeros((nfeat,1,maxlen), dtype=np.float32)

    with env.begin(write=True) as txn:
        for i in range(nTrain):
            print '\n', i+1, '/', nTrain

            Xmf[:] = 0

            vid_file = train_video_files[i]
            bn = os.path.basename(vid_file)
            prefix = os.path.splitext(bn)[0].strip('_rgb')

            vid_label = int(prefix[17:20]) - 1    # needs to be 0-based, since they are stored as 1-based

            nDepthImages = 0

            k = 0
            point_clouds = []
            while True:
                # Retrieve the corresponding depth image
                depth_file = os.path.join(depth_dir, prefix)
                depth_file = os.path.join(depth_file, 'MDepth-%08d.png' % (k+1))  # image file names are 1-based
                #depth_file = os.path.join(depth_file, 'Depth-%08d.png' % (k+1))  # image file names are 1-based
                if not os.path.exists(depth_file):
                    break  # end of sequence

                # Get the point cloud for this depth image (it's already normalized)
                pc = depth_to_pc(depth_file, N=NUM_POINT)
                point_clouds.append(pc.copy())

                nDepthImages += 1
                k += 1

            point_clouds = np.array(point_clouds).astype(np.float32)  # nDepthImages x NUM_POINT x 3

            num_batches = nDepthImages // BATCH_SIZE
            print 'Num files:', nDepthImages

            for batch_idx in range(num_batches):
                start_idx = batch_idx*BATCH_SIZE
                end_idx = min(nDepthImages, (batch_idx+1)*BATCH_SIZE)

                cur_batch_size = end_idx - start_idx
                if cur_batch_size <= 0:
                    break

                sample = point_clouds[start_idx:end_idx,:,:]
                feed_dict = {ops['pointclouds_pl']: sample, ops['is_training_pl']: is_training}
                feat_vec = sess.run(ops['feat'], feed_dict=feed_dict)
                Xmf[:,0,start_idx:end_idx] = feat_vec.T.copy()

            datum = caffe_pb2.Datum()
            datum.channels = nfeat
            datum.height = 1
            datum.width = nDepthImages
            datum.data = Xmf[:,0,:nDepthImages].tobytes()
            datum.label = vid_label
            str_id = '%09d' % i
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

            ofs.write('%d %s\n' % (i,prefix))

    ofs.close()


if __name__=='__main__':
    with tf.Graph().as_default():
        go()
