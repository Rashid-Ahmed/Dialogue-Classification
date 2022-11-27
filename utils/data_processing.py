import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch

def clean_soc(data):
  if data['soc'].dtype=='object':    
        y_soc = np.array(deepcopy(data['soc'])) 
        y_soc[y_soc == '35'] = 0
        y_soc[y_soc == '36'] = 1
        y_soc[y_soc == '37'] = 2
        y_soc[y_soc == '39'] = 3
        y_soc[y_soc == '41'] = 4
        y_soc[y_soc == '-1'] = 5
  else:
        y_soc = np.array(deepcopy(data['soc']))
        y_soc[y_soc == 35] = 0
        y_soc[y_soc == 36] = 1
        y_soc[y_soc == 37] = 2
        y_soc[y_soc == 39] = 3
        y_soc[y_soc == 41] = 4
        y_soc[y_soc == -1] = 5
  return data['message'], y_soc

def get_splitted_data(TYPE_MODEL, data):
    if TYPE_MODEL == 'self':
        return data[:, 0]
    elif TYPE_MODEL == 'parents':
        return data[:, 1]
    elif TYPE_MODEL == 'teacher':
        return data[:, 2]
    elif TYPE_MODEL == 're':
        return data[:, 3]
    elif TYPE_MODEL == 'cause':
        return data[:, 4]



def get_one_hot(offset, y_epi):
    one_hot = np.zeros((y_epi.shape[0], 5))
    for i in range(y_epi.shape[0]):
        off = offset + i
        if ("1" in y_epi[off] and "-1" not in y_epi[off]) or (1 == y_epi[off]):
            one_hot[i][0] = 1
        if "2" in y_epi[off] or 2 == y_epi[off]:
            one_hot[i][1] = 1
        if "3" in y_epi[off] or 3 == y_epi[off]:
            one_hot[i][2] = 1
        if "4" in y_epi[off] or 4 == y_epi[off]:
            one_hot[i][3] = 1
        if "5" in y_epi[off] or 5 == y_epi[off]:
            one_hot[i][4] = 1
    return one_hot

def get_split_tensors(vectors, one_hot_data, device):
    
    X_train, X_test, y_train, y_test = train_test_split(vectors, one_hot_data, test_size=0.2, random_state=0, shuffle = False)
    
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)
    
    return X_train, X_test, y_train, y_test