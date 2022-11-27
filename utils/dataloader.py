import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.data_processing import get_one_hot, clean_soc, get_splitted_data

class DiscourseDataset(Dataset):
  def __init__(self, csv_file, chunk_size, len_dataset, max_seq_len, vector_dim, embedding_model, target_type, embeddings_scale = 'word'):
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
    self.Type = embeddings_scale
    self.target_type = target_type
    self.len_dataset = len_dataset//chunk_size
    self.max_seq_len = max_seq_len
    self.vector_dim = vector_dim
    self.reader = pd.read_csv(self.csv_file, chunksize = self.chunk_size, dtype={'epi': 'str'})
    

    self.model = embedding_model
  def __len__(self):
    return self.len_dataset
  
  def __getitem__(self, index):
    offset = index * self.chunk_size
    batch = next(self.reader)
    batch = batch.reset_index()
    sentences = batch['message']
    if self.target_type != 'soc':
      targets = batch['epi']
      targets = get_one_hot(targets)
      targets, _ = get_splitted_data(self.target_type, targets)
    else:
      targets = clean_soc(batch)
    #Converting sentences into embedding vectors

    if self.Type == 'word':
        vector_embeddings = np.empty((self.chunk_size, self.max_seq_len, self.vector_dim))
        for i in range(self.chunk_size):
          message = sentences[i].split()
          for j in range(self.max_seq_len):
              if j < len(message):
                  vector_embeddings[i][j] = self.model.get_word_vector(message[j])
              else:
                  vector_embeddings[i][j] = 0
    
          vector_embeddings[i][self.max_seq_len - 1] = len(message)
          
    else:
        vector_embeddings = np.empty((self.chunk_size, self.vector_dim))
        for i in range(self.chunk_size):
          message = sentences[i]
          vector_embeddings[i] = self.model.get_sentence_vector (message)
         
    return vector_embeddings, targets

  
