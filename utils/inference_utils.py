import numpy as np
import torch
import json

def get_input_vector(sentence, embedding_model, EMBEDDING_SIZE, SENTENCE_SIZE, device):
  sentence = sentence.split()
  sentence_vector = np.empty((1, SENTENCE_SIZE, EMBEDDING_SIZE))
  for i in range (len(sentence)):
    if i < SENTENCE_SIZE:
      sentence_vector[0][i] = embedding_model.get_word_vector(sentence[i])
  sentence_vector[0][-1] = len(sentence)
  sentence_vector = torch.from_numpy(sentence_vector).float().to(device)
  return sentence_vector

def get_predictions_JSON(MODEL_TYPES, MODELS, sentence_vector):
  
  predictions = {}
  for i in range(len(MODELS)):
    outputs = MODELS[i](sentence_vector)
    _, prediction = torch.max(torch.FloatTensor(outputs.cpu()), 1)
    if i < 6:
      if prediction == 0:
        predictions[MODEL_TYPES[i]] = 'False'
      else:
        predictions[MODEL_TYPES[i]] = 'True'
    else:
      if prediction == 0:
        predictions[MODEL_TYPES[i]] = 'Externalization'
      elif prediction == 1:
        predictions[MODEL_TYPES[i]] = 'Elicitation'
      elif prediction == 2:
        predictions[MODEL_TYPES[i]] = 'Conflict'
      elif prediction == 3:
        predictions[MODEL_TYPES[i]] = 'Acceptence'
      elif prediction == 4:
        predictions[MODEL_TYPES[i]] = 'Integration'
      elif prediction == 5:
        predictions[MODEL_TYPES[i]] = 'None'
    
  json_predictions = json.dumps(predictions)

        
  return json_predictions
