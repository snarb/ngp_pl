import math
WORKERS_CNT = 16
WORKERS_CNT = 0
#USE_DDP = False
TEST_VIEW_ID = 15
TEST_VIEW_ID = 0
IM_W = 2704 // 2
#IM_W = 806
#IM_W = 540 --batch_size 16384
IM_H = 2028 // 2
#IM_H = 604
#IM_H = 405
VAL_BATCH_SIZE = 8192 * 20
VAL_BATCH_SIZE = IM_H * 80
VAL_BATCH_SIZE = IM_H * 25

BATCHED_EVAL = False
BATCHES_PER_EPOCH = 1000
BATCHES_PER_EPOCH = 300
BATCHES_PER_EPOCH = 30
GPU_CNT = 1
RAYS_CNT = IM_W * IM_H
VAL_BATCHES_PER_IMG = math.ceil(RAYS_CNT / VAL_BATCH_SIZE)
PROJECT_NAME =  'ngp_pl_scale_tuning_29_scale_NO_EXT'
VID_DIR = '/home/ubuntu/flame_steak/'
ATEN_FOLDER = '/home/ubuntu/flame_steak/temp/aten/'
MIN_FRAME = 0
MAX_FRAME = 30

FRAME_CHANGE_PROB = 1. / 100
#FRAME_CHANGE_PROB = 1. / 3
TEMP_DIR = '/home/ubuntu/flame_steak/temp/'