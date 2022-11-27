import pandas as pd
import torch
import os
from utils.get_embeddings import get_model
from utils.data_processing import get_one_hot, get_splitted_data
from utils.train_model import get_model_weights, initialize_model, train_model, load_existing_model

EMBEDDING_SIZE = 100
SENTENCE_SIZE = 79
DATA_DIR = 'data'
EMBEDDING_MODEL_NAME = 'cc.de.100.bin'
EPI_DATA_DIR = os.path.join(os.getcwd(), DATA_DIR, 'Augmented_Data_EPI.csv')
SOC_DATA_DIR = os.path.join(os.getcwd(), DATA_DIR, 'Augmented_Data_SOC.csv')
EPI_DATA_LEN = 16857
SOC_DATA_LEN = 17103
CHECKPOINT_DIR = os.path.join(os.getcwd(), 'checkpoints')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
embedding_model = get_model(os.path.join(DATA_DIR, EMBEDDING_MODEL_NAME))

NUM_LAYERS = 2
HIDDEN_SIZE = 400
STEP_SIZE =  0.0003

EPOCHS = 2
BATCH_SIZE = 512
OUTPUT_SIZE_EPI = 2
OUTPUT_SIZE_SOC = 6
MODEL_TYPES = ['self', 'parents', 'teacher', 're', 'cause', 'none', 'soc']


data_epi = pd.read_csv(EPI_DATA_DIR)
data_epi = get_one_hot(data_epi['epi'])
data_soc = None

def start_processes(TYPE_MODEL, data, DATA_FILE, DATA_LEN,load_model = False):
    data, OUTPUT_SIZE = get_splitted_data(TYPE_MODEL, data)
    model_weights = None
    if TYPE_MODEL!='soc':
        model_weights = get_model_weights(data)
    model, criterion, optimizer = initialize_model(device,OUTPUT_SIZE ,HIDDEN_SIZE , NUM_LAYERS, EMBEDDING_SIZE, STEP_SIZE, model_weights = model_weights)
    if load_model == True:
        model = load_existing_model(os.path.join('checkpoints', 'model_'+TYPE_MODEL+'.ckpt'), device, OUTPUT_SIZE_EPI, HIDDEN_SIZE , NUM_LAYERS, EMBEDDING_SIZE, STEP_SIZE)

    train_model(model, optimizer, criterion, TYPE_MODEL, BATCH_SIZE, EPOCHS, SENTENCE_SIZE, EMBEDDING_SIZE, embedding_model, DATA_FILE, CHECKPOINT_DIR, DATA_LEN, device)




# %%
#start_processes('parents', data_epi, EPI_DATA_DIR, EPI_DATA_LEN, load_model = False)
#start_processes('none', data_epi, EPI_DATA_DIR, EPI_DATA_LEN, load_model = False)
#start_processes('self', data_epi, EPI_DATA_DIR, EPI_DATA_LEN, load_model = False)
#start_processes('teacher', data_epi, EPI_DATA_DIR, EPI_DATA_LEN,  load_model = False)
#start_processes('re', data_epi, EPI_DATA_DIR, EPI_DATA_LEN, load_model = False)
#start_processes('cause', data_epi, EPI_DATA_DIR, EPI_DATA_LEN, load_model = False)
start_processes('soc', data_soc, SOC_DATA_DIR, SOC_DATA_LEN, load_model = False)



