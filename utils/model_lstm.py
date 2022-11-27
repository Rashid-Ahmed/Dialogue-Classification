import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


#WORD LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, output_size = 2, hidden_dim = 150, num_layers = 2, embedding_dim = 100, device = 'cpu'):
        
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, dropout = 0.25, num_layers = num_layers, batch_first=True, bidirectional = True)
        #batch_first = True means batch has dimension 0
        #input sequence to classifier is  (batchsize, padded sequence length, embedding size)

        self.fc = nn.Linear(hidden_dim, output_size)
        #self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):
        
        seq_lengths = input[: , -1 , 0]
        pack = pack_padded_sequence(input, seq_lengths.to('cpu'), enforce_sorted = False, batch_first=True).to(self.device)
        h0 = torch.ones(self.num_layers * 2, input.size(0), self.hidden_dim).to(self.device)
        #h0 is the first hidden state(at time step 0) for the lstm network and input.size(0) is our batch size
        c0 = torch.ones(self.num_layers * 2, input.size(0), self.hidden_dim).to(self.device)
        #c0 is the first cell state(at time step 0) for the lstm network (lstms have cells as well unlike gru or rnn which is why we need this c0)
   
        hidden, (ht, ct) = self.lstm(pack, (h0, c0))
        #out is of shape(batch_size, seq_length, hidden_size))
        # batch_size means the output for each sentence in a batch
        # seq_length means that for each time step we get a output(i.e for each word we get a output) however we only need the last output after the last word
        # hidden_size this is for the output features of the LSTM layers we feed this to our linear layer to get our number of outputs
        out = self.fc(ht[-1])
        #out = self.fc(hidden[:, -1, :])
        #out = self.softmax(out)
        # this is how we take only the last time steps features for all the batches.
        return out
		    


    