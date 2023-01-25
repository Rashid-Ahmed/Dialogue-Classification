
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.data_processing import get_one_hot, clean_soc, get_splitted_data

class DiscourseDataloader(Dataset):
  def __init__(self, csv_file, chunk_size, MODEL, len_dataset, target_type):
    """
    Args:
        csv_file (string): Path to the csv file.
        chunk_size: Batch size
        len_dataset: total number of rows in the dataset
        max_seq_len: maximum length of sequence(maximum length of words in a sentence allowed, anymore will be clipped)
        vector_dim: Dimension of our word embeddings
        train(optional): Do we train our embedding Model
        transform (optional): Optional transform to be applied
            on a sample.
    """
    self.csv_file = csv_file
    self.chunk_size = chunk_size
    self.target_type = target_type
    self.len_dataset = len_dataset//chunk_size
    self.tokenizer =  AutoTokenizer.from_pretrained(MODEL)
    self.reader = pd.read_csv(self.csv_file, chunksize = self.chunk_size)
    

  def __len__(self):
    return self.len_dataset
  
  def __getitem__(self, index):
    offset = index * self.chunk_size
    batch = next(self.reader)
    if self.target_type != 'soc':
      targets = batch['epi']
      sentences = batch['message'].tolist()
    else:
      sentences, targets = clean_soc(batch)
      sentences = sentences.tolist()
    #Converting sentences into embedding vectors
    
    embeddings = self.tokenizer(sentences, padding=True, truncation=True, max_length = 65, return_tensors='pt')
    return embeddings, targets

  



  
