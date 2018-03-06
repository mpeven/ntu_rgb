import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing
import os, sys

# def expand_file(file_name):
#     ra = np.load(file_name)['arr_0']
#     new_file = file_name.replace('2D', '2D_npy').replace('npz', 'npy')
#     np.save(new_file, ra)
#     return ""
#
# all_files = []
# for x in range(56880):
#     if os.path.isfile('/hdd/Datasets/NTU/nturgb+d_op_flow_2D_npy/{:05}.npy'.format(x)):
#         continue
#     all_files.append('/hdd/Datasets/NTU/nturgb+d_op_flow_2D/{:05}.npz'.format(x))
#
# with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
#     list(tqdm(p.imap(expand_file, all_files), total=len(all_files)))

# vid_id = 0
# x = np.load('/hdd/Datasets/NTU/nturgb+d_op_flow_2D_npy/{:05}.npy'.format(vid_id))
# print(x.shape)



zeros = np.zeros([50,400,400,1])

PNG_CACHE = '/hdd/Datasets/NTU/nturgb+d_op_flow_2D_png'
OP_FLOW_CACHE = '/hdd/Datasets/NTU/nturgb+d_op_flow_2D'

def create_ims_from_op_flow(vid_id):
    # Make dir if it doesn't exist
    if not os.path.exists('{}/{:05}'.format(PNG_CACHE, vid_id)):
        os.makedirs('{}/{:05}'.format(PNG_CACHE, vid_id))

    # Load optical flow
    op_flow_orig = np.load('{}/{:05}.npz'.format(OP_FLOW_CACHE, vid_id))['arr_0']

    # Rescale to integer
    m0 = np.min(op_flow_orig)
    m1 = np.max(op_flow_orig)
    rescaled = 255 * (op_flow_orig + np.abs(m0))/(m1-m0)

    # Save max and min
    np.save('{}/{:05}/min_max'.format(PNG_CACHE, vid_id), np.array([m0, m1]))

    # Add third channel
    op_flow_as_ims = rescaled.reshape([50,400,400,2]).astype(np.uint8)
    op_flow_rgb = np.concatenate([op_flow_as_ims, zeros], 3).astype(np.uint8)

    # Save images
    for i in range(50):
        im = cv2.imwrite('{}/{:05}/{:02}.png'.format(PNG_CACHE, vid_id, i), op_flow_rgb[i])


with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    list(tqdm(p.imap(create_ims_from_op_flow, range(56880)), total=56880))

# for vid_id in tqdm(range(56880)):
#     op_flow = np.zeros([50, 400, 400, 2])
#     for i in range(50):
#         im = cv2.imread('{}/{:05}/{:02}.png'.format(PNG_CACHE, vid_id, i))
#         op_flow[i,:,:,0] = im[:,:,0]
#         op_flow[i,:,:,1] = im[:,:,1]
#
#     feature = op_flow.reshape([5,20,400,400])
#
#     # Rescale
#     m0, m1 = np.load('{}/{:05}/min_max.npy'.format(PNG_CACHE, vid_id))
#     feature = ((feature/255.)*(m1-m0))-np.abs(m0)
