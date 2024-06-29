import torch
import torch.nn as nn

class RefinedRNN(nn.Module):
    """
    A Refined RNN model with dropout and batch normalization.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, seq_length, device):
        super(RefinedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(seq_length)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = self.bn(out)
        out = self.fc(out[:, -1, :])
        return out
