import numpy as np
import torch
import json

def get_predictions_JSON(MODEL_TYPES, MODELS, TOKENIZER, sentence, device):
  
  predictions = {}
  for i in range(len(MODELS)):
    vectors = TOKENIZER(sentence, padding=True, truncation=True, max_length = 65, return_tensors='pt')
    vectors = vectors.to(device)
    outputs = MODELS[i](**vectors).logits
    _, prediction = torch.max(torch.FloatTensor(outputs.detach().cpu()), 1)
    if MODEL_TYPES[i]!='soc' and MODEL_TYPES[i]!='SOC':
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
