from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.NAME = 'ActivityNetCaptions'
_C.DATA.DATA_PATH = ''
_C.DATA.ANNO_PATH = 'data/anet'
_C.DATA.NUM_FRAME_PATH = 'data/anet/anet_num_frame.json'
_C.DATA.VOCAB_PATH = ''
_C.DATA.NUM_SAMPLE_FRAME = 32
_C.DATA.WINDOW_SIZE = 16
_C.DATA.SAMPLING_RATE = 1
_C.DATA.NUM_MAX_WORD = 30
_C.DATA.WORD_EMBEDDING_DIM = 300
_C.DATA.NUM_SEN_PER_VIDEO = 4
_C.DATA.NUM_MAX_CLIP = 40
_C.DATA.VOCAB_SIZE = 2
_C.DATA.MLM_P = 0.0
# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224
_C.DATA.INPUT_SIZE = 224
# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 12

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = False

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False




_C.BACKBONE = CN()
# Model architecture.
_C.BACKBONE.ARCH = "slowfast"

_C.BACKBONE.FREEZE = True
# Model name
_C.BACKBONE.MODEL_NAME = "SlowFast"
_C.BACKBONE.PRETRAINED = ""


# Model architectures that has one single pathway.
_C.BACKBONE.SINGLE_PATHWAY_ARCH = [
    "2d",
    "c2d",
    "i3d",
    "slow",
    "x3d",
    "mvit",
]

# Model architectures that has multiple pathways.
_C.BACKBONE.MULTI_PATHWAY_ARCH = ["slowfast"]


_C.SLOWFAST = CN()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 4

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.TRANSFORM = 'clip'
_C.TRAIN.CHECKPOINT_FILE_PATH = ''
_C.TRAIN.EVAL_PERIOD = 1
_C.TRAIN.SAVE_CHECKPOINT = False
_C.TRAIN.CHECKPOINT_PERIOD = 10
_C.TRAIN.MIXED_PRECISION = False

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.IOU = [0.5, 0.7]
_C.TEST.EVAL_TRAIN = False
_C.TEST.TRANSFORM = 'clip'
_C.TEST.CHECKPOINT_FILE_PATH = ""

_C.MODEL = CN()
_C.MODEL.NAME = 'ClipByClip'
_C.MODEL.MODEL_DIM = 512
_C.MODEL.NUM_CROSS_LAYER = 2
_C.MODEL.DROPOUT = 0.0
_C.MODEL.DROPATT = 0.0
_C.MODEL.NUM_HEAD = 4
_C.MODEL.INNER_DIM = 1024
_C.MODEL.MEM_LEN = 3
_C.MODEL.VIDENCODER_LAYER = 3
_C.MODEL.NUM_LAYER = 3
_C.MODEL.C_MEM_LEN = 3
_C.MODEL.COMPRESS_TYPE = 'conv'
_C.MODEL.COMPRESS_RATE = 4
_C.MODEL.NUM_BUCKETS = 32
_C.MODEL.BIDIRECTIONAL = False

_C.LOSS = CN()
_C.LOSS.NAME = 'two_step_loss'
_C.LOSS.BETA = 0.1
_C.LOSS.LAMBDAS = []
_C.LOSS.CLIP_GT_THRES = 0.5
# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.GRAD_ACC_STEP = 1
# Base learning rate.
_C.SOLVER.BASE_LR = 4e-4

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 12

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-6
# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 2

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 4e-5

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "adam"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = True

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = 1.0

# LARS optimizer
_C.SOLVER.LARS_ON = False

_C.SOLVER.PARAM_GROUP_PREFIX = []

_C.SOLVER.PARAM_GROUP_LR = []
# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


_C.DDP = CN()
_C.DDP.NUM_GPUS = 1
_C.DDP.INIT_METHOD = "tcp://localhost:9999"
_C.DDP.DIST_BACKEND = "nccl"
_C.DDP.NUM_SHARDS = 1
_C.DDP.SHARD_ID = 0

_C.TENSORBOARD = CN()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""


_C.RNG_SEED = 1
_C.OUTPUT_DIR = "."
_C.LOG_DIR = "log"
_C.LOG_PERIOD = 50
_C.COMMENT = 'test eval code'