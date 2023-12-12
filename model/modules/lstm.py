import torch.nn as nn
from .decoder import Decoder

class LSTMModule(nn.Module):
    def __init__(self, input_size, sequence_length, lstm_hidden_size):
        super(LSTMModule, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers=self.num_layers, batch_first=True)

    def forward(self, x):
        emb, _ = self.lstm(x)
        return emb

class LSTMPred(nn.Module):
    def __init__(self, input_size, sequence_length, lstm_hidden_size, output_size=1):
        super(LSTMPred, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = LSTMModule(input_size, sequence_length, lstm_hidden_size)
        self.decoder = Decoder(lstm_hidden_size, output_size)

    def forward(self, x):
        emb = self.lstm(x)
        output = self.decoder(emb[:, -1, :])
        return output