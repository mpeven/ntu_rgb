'''
Server & Model configuration parameters
'''
# from models import Model_1, Model_2, Model_3, Model_4, Model_5

#### Set these:
EXPERIMENT_NUM = 6
MACHINE_NUM = 1 # 0 - mac, 1 - Titan, 2 - MARCC, 3 - MCEH
NUM_WORKERS = 4 # Cores
####

#### For SYSU:
SPLIT_NUMBER = 0
####


##################################################
# Cache directories

### Mac
if MACHINE_NUM == 0:
    CACHE_METADATA      = '/Users/mpeven/Projects/Activity_Recognition/cache/metadata.pickle'
    CACHE_3D_OP_FLOW    = '/Users/mpeven/Projects/Activity_Recognition/cache/optical_flow_3D'
    CACHE_RGB_VID       = '/Users/mpeven/Projects/Activity_Recognition/cache/nturgb+d_rgb'
    CACHE_IR_VID        = '/Users/mpeven/Projects/Activity_Recognition/cache'
    CACHE_DEPTH         = '/Users/mpeven/Projects/Activity_Recognition/cache'
    CACHE_MASKED_DEPTH  = '/Users/mpeven/Projects/Activity_Recognition/cache/nturgb+d_depth_masked'
    CACHE_SKELETONS     = '/Users/mpeven/Projects/Activity_Recognition/cache'
    SYSU_LOCATION       = '/Users/mpeven/Downloads/SYSU3DAction'


### Titan
if MACHINE_NUM == 1:
    CACHE_METADATA       = '/home/mike/Documents/Activity_Recognition/cache/metadata.pickle'
    CACHE_RGB_VID        = '/hdd/Datasets/NTU/nturgb+d_rgb'
    CACHE_2D_IMAGES      = '/hdd/Datasets/NTU/nturgb+d_rgb_masked'
    CACHE_3D_IMAGES      = '/hdd/Datasets/NTU/ntu_3D_voxel_images'
    CACHE_2D_OP_FLOW     = '/hdd/Datasets/NTU/nturgb+d_op_flow_2D_npy'
    CACHE_2D_OP_FLOW_PNG = '/hdd/Datasets/NTU/nturgb+d_op_flow_2D_png'
    CACHE_IR_VID         = '/hdd/Datasets/NTU/nturgb+d_ir'
    CACHE_DEPTH          = '/hdd/Datasets/NTU/nturgb+d_depth'
    CACHE_MASKED_DEPTH   = '/hdd/Datasets/NTU/nturgb+d_depth_masked'
    CACHE_SKELETONS      = '/hdd/Datasets/NTU/nturgb+d_skeletons'
    CACHE_3D_OP_FLOW     = '/hdd/Datasets/NTU/nturgb+d_op_flow_3D'
    CACHE_FEATURES_VOX_FLOW = '/home/mike/Documents/Activity_Recognition/nturgb+d_features_small'

    SYSU_LOCATION          = '/home/mike/Documents/SYSU'
    CACHE_3D_VOX_FLOW_SYSU = '/home/mike/Documents/Activity_Recognition/SYSU_voxel_flow_3D_54'
    CACHE_2D_IMAGES_SYSU   = '/home/mike/Documents/Activity_Recognition/SYSU_rgb_images_5_npy'

### MARCC
if MACHINE_NUM == 2:
    CACHE_METADATA     = '/home-3/mpeven1@jhu.edu/work/dev_mp/nturgb_cache/metadata.pickle'
    CACHE_RGB_VID      = '/home-3/mpeven1@jhu.edu/data/nturgb+d_rgb'
    CACHE_IR_VID       = '/home-3/mpeven1@jhu.edu/data/nturgb+d_ir'
    CACHE_DEPTH        = '/home-3/mpeven1@jhu.edu/data/nturgb+d_depth'
    CACHE_MASKED_DEPTH = '/home-3/mpeven1@jhu.edu/data/nturgb+d_depth_masked'
    CACHE_SKELETONS    = '/home-3/mpeven1@jhu.edu/data/nturgb+d_skeletons'
    CACHE_3D_OP_FLOW   = '/home-3/mpeven1@jhu.edu/work/dev_mp/nturgb_cache/optical_flow_3D'

### GPU Server
if MACHINE_NUM == 3:
    CACHE_METADATA     = '/home/mpeven1/rambo/home/mpeven/ntu_rgb/cache/metadata.pickle'
    CACHE_RGB_VID      = '/home/mpeven1/rambo/edata/nturgb/nturgb+d_rgb'
    CACHE_IR_VID       = '/home/mpeven1/rambo/edata/nturgb/nturgb+d_ir'
    CACHE_DEPTH        = '/home/mpeven1/rambo/edata/nturgb/nturgb+d_depth'
    CACHE_MASKED_DEPTH = '/home/mpeven1/rambo/edata/nturgb/nturgb+d_depth_masked'
    CACHE_SKELETONS    = '/home/mpeven1/rambo/edata/nturgb/nturgb+d_skeletons'




##################################################
# Experiments

### Default
NUM_EPOCHS = 50
HIDDEN_DIM_SIZE = 256
LSTM_DROPOUT = 0

EXPERIMENTS = [
    {       ### 1 ###
        'images': True,
        'batch_size': 32,
    }, {    ### 2 ###
        'images': True,
        'batch_size': 32,
        'cross_view': True,
    }, {    ### 3 ###
        'images_3D': True,
        'batch_size': 8,
    }, {    ### 4 ###
        'images_3D': True,
        'batch_size': 8,
        'cross_view': True,
    }, {    ### 5 ###
        'op_flow': True,
        'batch_size': 8,
    }, {    ### 6 ###
        'op_flow': True,
        'batch_size': 8,
        'cross_view': True,
    }, {    ### 7 ###
        'op_flow': True,
        'batch_size': 8,
        'augmentation': False,
    }, {    ### 8 ###
        'op_flow': True,
        'batch_size': 8,
        'augmentation': False,
        'cross_view': True,
    }, {    ### 9 ###
        'op_flow_2D': True,
        'batch_size': 24,
    }, {    ### 10 ###
        'op_flow_2D': True,
        'batch_size': 16,
        'cross_view': True,
    }, {    ### 11 ###
        'op_flow_2D': True,
        'batch_size': 128,
    }, {    ### 12 ###
        'op_flow_2D': True,
        'batch_size': 128,
        'cross_view': True,
    }, {    ### 13 ###
        'op_flow': True,
        'batch_size': 8,
        'dataset': 'SYSU',
        'dataset_classes': 12,
    }, {    ### 14 ###
        'images': True,
        'batch_size': 32,
        'dataset': 'SYSU',
        'dataset_classes': 12,
    },
]

# Set defaults
for experiment in EXPERIMENTS:
    if 'dataset' not in experiment:
        experiment['dataset'] = 'NTU'
        experiment['dataset_classes'] = 60
    if 'images' not in experiment:
        experiment['images'] = False
    if 'images_3D' not in experiment:
        experiment['images_3D'] = False
    if 'op_flow' not in experiment:
        experiment['op_flow'] = False
    if 'op_flow_2D' not in experiment:
        experiment['op_flow_2D'] = False
    if 'augmentation' not in experiment:
        experiment['augmentation'] = True
    if 'cross_view' not in experiment:
        experiment['cross_view'] = False
    if 'single_features' not in experiment:
        experiment['single_feature'] = False

# Dataset config
DATASET             = EXPERIMENTS[EXPERIMENT_NUM-1]['dataset']
DATASET_NUM_CLASSES = EXPERIMENTS[EXPERIMENT_NUM-1]['dataset_classes']

# Experiment config
DATA_IMAGES       = EXPERIMENTS[EXPERIMENT_NUM-1]['images']
DATA_IMAGES_3D    = EXPERIMENTS[EXPERIMENT_NUM-1]['images_3D']
DATA_OP_FLOW      = EXPERIMENTS[EXPERIMENT_NUM-1]['op_flow']
DATA_OP_FLOW_2D   = EXPERIMENTS[EXPERIMENT_NUM-1]['op_flow_2D']
DATA_AUGMENTATION = EXPERIMENTS[EXPERIMENT_NUM-1]['augmentation']
DATA_CROSS_VIEW   = EXPERIMENTS[EXPERIMENT_NUM-1]['cross_view']
DATA_BATCH_SIZE   = EXPERIMENTS[EXPERIMENT_NUM-1]['batch_size']
DATA_SINGLE_FEAT  = EXPERIMENTS[EXPERIMENT_NUM-1]['single_feature']

def print_config():
    data_desc = ""
    if DATA_IMAGES:
        data_desc += "2D images"
    if DATA_IMAGES_3D:
        data_desc += "3D images"
    if DATA_OP_FLOW:
        data_desc += "3D optical flow"
    if DATA_OP_FLOW_2D:
        data_desc += "2D optical flow"
    if DATA_SINGLE_FEAT:
        data_desc += " single feature"
    if not DATA_AUGMENTATION:
        data_desc += " (no data augmentation)"
    if DATA_CROSS_VIEW:
        data_desc += " - cross-view split"
    else:
        data_desc += " - cross-subject split"
    if DATASET == "SYSU":
        data_desc += " {:02}".format(SPLIT_NUMBER)

    print("Experiment {:02} - {} - {}".format(EXPERIMENT_NUM, data_desc, DATASET))

from models import *
if EXPERIMENT_NUM in [1,2]:
    NEURAL_NET = Model_1()
elif EXPERIMENT_NUM in [3,4]:
    NEURAL_NET = Model_4()
elif EXPERIMENT_NUM in [5,6,7,8]:
    NEURAL_NET = Model_2()
elif EXPERIMENT_NUM in [9,10]:
    NEURAL_NET = Model_5()
elif EXPERIMENT_NUM in [11,12]:
    NEURAL_NET = Model_5_small()
elif EXPERIMENT_NUM in [13]:
    NEURAL_NET = Model_2()
elif EXPERIMENT_NUM in [14]:
    NEURAL_NET = Model_1()
