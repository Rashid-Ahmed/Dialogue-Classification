import pandas as pd
import torch
import os
from utils.train_transformer import load_existing_model
from utils.get_embeddings import get_model
from utils.networking import start_tcp_server
from transformers import AutoTokenizer

SENTENCE_SIZE = 79
EMBEDDING_SIZE = 100
CHECKPOINT_DIR = 'checkpoints'
MODEL_NAME = 'xlm-roberta-base'
MODEL_TYPES = ['self', 'parents', 'teacher', 're', 'cause', 'soc']
TOKENIZER =  AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

MODELS = []
for i in range(len(MODEL_TYPES)):
  if MODEL_TYPES[i] == 'soc':
    OUTPUT_SIZE = 6
  else:
    OUTPUT_SIZE = 2
  MODELS.append(load_existing_model(os.path.join(CHECKPOINT_DIR, 'model_'+MODEL_TYPES[i]+'.ckpt'), MODEL_NAME, OUTPUT_SIZE, device))

start_tcp_server('localhost', 10000, MODEL_TYPES, MODELS, TOKENIZER, device)


