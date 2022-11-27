from json import load
from mimetypes import init
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.model_lstm import LSTMClassifier
from utils.dataloader import DiscourseDataset
from utils.selection import model_confidence
import copy 

def train_model(model, optimizer, criterion, TYPE_MODEL, BATCH_SIZE, EPOCHS, SENTENCE_SIZE, EMBEDDING_SIZE, EMBEDDING_MODEL, DATA_FILE, CHECKPOINT_DIR, DATA_LEN, device):
    BEST_ACC = 0
    if TYPE_MODEL!= 'soc':
        accuracy_0 = 0.7
        accuracy_1 = 0.7
    else:
        accuracy_soc = 0.55
    for epoch in range(EPOCHS):
        print (epoch)
        epoch_loss = 0
        test_predictions = []
        test_truth = []
        #Initializing Custom Pytorch Dataset class
        dataset = DiscourseDataset(csv_file = DATA_FILE, chunk_size = BATCH_SIZE, len_dataset = DATA_LEN, max_seq_len = SENTENCE_SIZE, vector_dim = EMBEDDING_SIZE, embedding_model = EMBEDDING_MODEL, 
                                   target_type = TYPE_MODEL, embeddings_scale = 'word')
        train_batches = int(len(dataset) * 0.8)

        for i in range(len(dataset)): 
          vectors, targets = dataset[i]
      
          
          embeddings = torch.from_numpy(vectors).float().to(device)
          labels = torch.from_numpy(targets).long().to(device)
 
          if i < train_batches:
            #Training the model
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += outputs.shape[0] * loss.item()
          else:
            #Saving test results
            outputs = model(embeddings)
            _, predictions = torch.max(torch.FloatTensor(outputs.cpu()), 1)
            test_predictions.extend(predictions.tolist())
            test_truth.extend(labels.cpu().tolist())
        
        #Printing test results and saving the model if the current test results are better than before
        test_predictions = np.array(test_predictions)
        test_truth = np.array(test_truth)
        test_confusion = confusion_matrix(test_truth, test_predictions)
        print(test_confusion)
        if TYPE_MODEL != 'soc':
            acc_0 =  (test_confusion[0][0]/(test_confusion[0][0] + test_confusion[0][1])) 
            acc_1 =  (test_confusion[1][1]/(test_confusion[1][0] + test_confusion[1][1]))

            if acc_0 > accuracy_0 and acc_1 >accuracy_1:
                accuracy_0 = acc_0
                accuracy_1 = acc_1
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_'+TYPE_MODEL+'.ckpt'))
                print(confusion_matrix(test_truth, test_predictions))
                print (acc_0, acc_1)
        else:
            acc = (test_truth==test_predictions).sum()/len(test_truth)
            if acc > accuracy_soc:
                accuracy_soc = acc
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_'+TYPE_MODEL+'.ckpt'))
                print (accuracy_soc)

          
        
            
    return model, BEST_ACC
  
def initialize_model(device, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_SIZE, STEP_SIZE, model_weights = None):
    
    
    model = LSTMClassifier(output_size = OUTPUT_SIZE, hidden_dim = HIDDEN_SIZE, num_layers = NUM_LAYERS, embedding_dim = EMBEDDING_SIZE, device = device).to(device)
    if model_weights == None:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight = model_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=STEP_SIZE)
    return model, criterion, optimizer

def load_existing_model(MODEL_PATH, device, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_SIZE):
    
    model = LSTMClassifier(output_size = OUTPUT_SIZE, hidden_dim = HIDDEN_SIZE, num_layers = NUM_LAYERS, embedding_dim = EMBEDDING_SIZE, device = device).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location = torch.device(device)))
    return model

def get_model_weights(data):
    weights = [1, (data == 0).sum()/(data == 1).sum()]
    model_weights = torch.FloatTensor(weights)
    return model_weights