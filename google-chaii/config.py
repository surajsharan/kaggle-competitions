import os
import pathlib
import torch
import numpy as np
import random
import multiprocessing
import wandb



def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value


def wand_config(config=None):
    w_config={}
    for name ,values in vars(config).items():
        if name.isupper():
            w_config[name] = values  
    return w_config


try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False


# Tracking
EXPERIMENT_NAME = "xlm_roberta_my"
CHANGE ="changed model with multi dropout model xlm roberta new cleaned data learning rate decreased"
RUN_PARALLEL = True




PATH = str(pathlib.Path().absolute())+''.join('/output/'+EXPERIMENT_NAME)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
SEED = fix_all_seeds(1993)
os.environ["CUDA_VISIBLE_DEVICES"]= "2,4,8,11,12,14"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DEVICE = torch.device('cuda:3')

NUM_WORKERS = 0 #optimal_num_of_loader_workers()
FP16 = False  #True if APEX_INSTALLED else False
FP16_OPT_LEVEL = "O1"


# # Model
MODEL_TYPE = "xlm_roberta"
MODEL_PATH = "xlm-roberta/deepset/xlm-roberta-large-squad2/"
CONFIG_NAME = "xlm-roberta/deepset/xlm-roberta-large-squad2/"

# MODEL_TYPE = "xlm_roberta-longformer"
# MODEL_PATH = "Peltarion/xlm-roberta-longformer-base-4096/"
# CONFIG_NAME = "Peltarion/xlm-roberta-longformer-base-4096/"





# MODEL_TYPE = "rembert"
# MODEL_PATH = "Rembert/google/rembert/"
# CONFIG_NAME = "Rembert/google/rembert/"

MODEL_OUT_NAME = "model.bin"

# Tokeinzer
# TOKENIZER = "Rembert/google/rembert/"
TOKENIZER = "xlm-roberta/deepset/xlm-roberta-large-squad2/"
# TOKENIZER = "Peltarion/xlm-roberta-longformer-base-4096/"

N_LAST_HIDDEN = 12



MAX_SEQ_LEN = 400
DOC_STRIDE = 192
GRADIENT_ACC_STEPS = 1
EARLY_STOPPING = 3

# Training
TRAINING_FILE = "input/train.csv"
EPOCHS = 10
TRAIN_BATCH_SIZE = 36
VALID_BATCH_SIZE = 128
LOGGING_STEPS = 4


# Optimizer
OPTIMIZER = 'AdamW'
LEARNING_RATE = 3e-5 # 1.5e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8
MAX_GRAD_NORM = 1.0

# Scheduler
DECAY_NAME =  'cosine-warmup'
WARMUP_RATIO = 0.1
