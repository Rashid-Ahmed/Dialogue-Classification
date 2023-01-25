import torch
import pandas as pd
import os
import warnings
from utils.train_transformer import train_model

BATCH_SIZE = 32
EPOCHS = 2
STEP_SIZE = 0.0000002
EPS = .00000001 
WEIGHT_DECAY = 0
MODEL = 'xlm-roberta-base'
TEST_LEN = 1081
DATA_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'
MODEL_TYPES = ['self', 'parents', 'teacher', 're', 'cause']
OUTPUTS_EPI = 2
OUTPUTS_SOC = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def start_processes(MODEL_TYPES, TEST_LEN, CHECKPOINT_DIR, device, DATA_DIR, EPOCHS, STEP_SIZE, BATCH_SIZE, WEIGHT_DECAY, EPS, OUTPUTS_EPI, OUTPUTS_SOC, train_type, load_epi_model = False):
    
    model_weights = None
    #Train EPI models
    if train_type == 'both' or train_type == 'EPI':
        for model_type in MODEL_TYPES:
            EPI_DATA_DIR = os.path.join(DATA_DIR, model_type)
            TRAIN_LEN = len(pd.read_csv(EPI_DATA_DIR+'_train.csv')) 
            train_model(OUTPUTS_EPI, MODEL, model_type, TRAIN_LEN, TEST_LEN, CHECKPOINT_DIR, device, EPI_DATA_DIR,EPOCHS, STEP_SIZE, BATCH_SIZE, WEIGHT_DECAY, EPS, model_weights, load_model = load_epi_model)
    
    if train_type == 'both' or train_type == 'SOC':
        #Train SOC models
        SOC_DATA_DIR = os.path.join(DATA_DIR, 'Data_SOC')
        TRAIN_LEN = len(pd.read_csv(SOC_DATA_DIR+'_train.csv')) 
        model_type = 'soc'
        train_model(OUTPUTS_SOC, MODEL, model_type, TRAIN_LEN, TEST_LEN, CHECKPOINT_DIR, device, SOC_DATA_DIR, EPOCHS, STEP_SIZE, BATCH_SIZE, WEIGHT_DECAY, EPS, model_weights, load_model = load_epi_model)



start_processes(MODEL_TYPES, TEST_LEN, CHECKPOINT_DIR, device, DATA_DIR, EPOCHS, STEP_SIZE, BATCH_SIZE, WEIGHT_DECAY, EPS, OUTPUTS_EPI, OUTPUTS_SOC, train_type = 'both', load_epi_model = False)