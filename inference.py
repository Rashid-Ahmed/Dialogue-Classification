from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import pandas as pd
import os
import torch
import copy
import numpy as np

DATA_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'
MODEL = 'xlm-roberta-base'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels = 6).to(device)
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'model_deberta34.ckpt'), map_location=device))
#tokenizer = AutoTokenizer.from_pretrained(MODEL)
