from sklearn.metrics import confusion_matrix
import torch
import numpy as np

def model_confidence(confidence_score, outputs, y_test_new):
    count = 0
    Outputs = []
    for i in range(len(outputs)):
        if (outputs[i] > confidence_score).sum() > 0:
            Outputs.append(outputs[i].tolist())
        else:
          #  print (len(Outputs))
            y_test_new[i] = -2
            count = count + 1
    y_test_new = y_test_new[y_test_new != -2]

    _, predictions_new = torch.max(torch.FloatTensor(Outputs), 1)
    predictions_new = np.array(predictions_new)
    print (confusion_matrix(y_test_new, predictions_new))
    print ((y_test_new == predictions_new).sum()/len(y_test_new))