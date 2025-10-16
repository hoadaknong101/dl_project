import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h0=None, c0=None):
        # Embedding layer
        x = self.embedding(x)

        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim,
                             x.size(0),
                             self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim,
                             x.size(0),
                             self.hidden_dim).to(x.device)


        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out, hn, cn