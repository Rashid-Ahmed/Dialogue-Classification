# %%
import pandas as pd
import fasttext
from scipy.spatial.distance import cosine
import numpy as np

def word_embeddings(embedding_file, model_train, VECTOR_DIM, CHUNK_SIZE):
    embedding_data = pd.read_csv(embedding_file)
    data_length = len(embedding_data)
    vector_embeddings = np.empty((data_length, CHUNK_SIZE, VECTOR_DIM))
    for i in range(data_length):
        words = embedding_data['message'][i].split()
        for j in range(CHUNK_SIZE):
            if j < len(words):
                vector_embeddings[i][j] = model_train.get_word_vector(words[j])
            else:
                vector_embeddings[i][j] = 0
        vector_embeddings[i][CHUNK_SIZE - 1] = len(words)
    return vector_embeddings

def sentence_embeddings(embedding_file, model_train, VECTOR_DIM):
    embedding_data = pd.read_csv(embedding_file)
    data_length = len(embedding_data)
    vector_embeddings = np.empty((data_length,VECTOR_DIM))
    for i in range(data_length):
        vector_embeddings[i] = model_train.get_sentence_vector(embedding_data['message'][i])
    return vector_embeddings

def vector_embeddings(embedding_file, embedding_model , VECTOR_DIM = 300, CHUNK_SIZE = 20, Word_embeddings = True, Ws = 5):

    if Word_embeddings == True:
        word_vector_embeddings = word_embeddings(embedding_file, embedding_model, VECTOR_DIM, CHUNK_SIZE)
        return word_vector_embeddings
    else:
        sentence_vector_embeddings = sentence_embeddings(embedding_file, embedding_model, VECTOR_DIM)
        return sentence_vector_embeddings

def test_time_embedding(text, model_type, embedding_type, CHUNK_SIZE, VECTOR_DIM):
    if model_type == 'pretrained':
        model_train = fasttext.load_model('cc.de.300.bin')
    else:
        model_train = fasttext.load_model('trained_model.bin')
    
    if embedding_type == 'word' or 'Word':
        message = text.split()
        vector_embeddings = np.empty((1, CHUNK_SIZE, VECTOR_DIM))

        for j in range(CHUNK_SIZE):
            if j < len(message):
                vector_embeddings[0][j] = model_train.get_word_vector(message[j])
            else:
                vector_embeddings[0][j] = 0

        vector_embeddings[0][CHUNK_SIZE - 1] = len(message)


    else:
        vector_embeddings = model_train.get_sentence_vector(text)
    
    return vector_embeddings

def get_model(model_path):
    model_train = fasttext.load_model(model_path)
    return model_train


