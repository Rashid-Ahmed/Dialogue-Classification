import pandas as pd
import torch
import os
from utils.train_transformer import load_existing_model
from utils.get_embeddings import get_model
from utils.networking import start_tcp_server
from transformers import AutoTokenizer
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

SENTENCE_SIZE = config.get('DEFAULT', 'SENTENCE_SIZE')
EMBEDDING_SIZE = config.get('DEFAULT', 'EMBEDDING_SIZE')
CHECKPOINT_DIR = config.get('DEFAULT', 'CHECKPOINT_DIR')
MODEL_NAME = config.get('DEFAULT', 'MODEL_NAME')
MODEL_TYPES = config.get('DEFAULT', 'MODEL_TYPES')
TOKENIZER =  config.get('DEFAULT', 'TOKENIZER')
DEVICE = config.get('DEFAULT', 'DEVICE')
IP = config.get('DEFAULT', 'IP')
PORT = config.get('DEFAULT', 'PORT')

MODELS = []
for i in range(len(MODEL_TYPES)):
  if MODEL_TYPES[i] == 'soc':
    OUTPUT_SIZE = 6
  else:
    OUTPUT_SIZE = 2
  MODELS.append(load_existing_model(os.path.join(CHECKPOINT_DIR, 'model_'+MODEL_TYPES[i]+'.ckpt'), MODEL_NAME, OUTPUT_SIZE, DEVICE))

start_tcp_server(IP, PORT, MODEL_TYPES, MODELS, TOKENIZER, DEVICE)


