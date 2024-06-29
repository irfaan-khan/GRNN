import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    A Simple LSTM model.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, seq_length, device):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            num_layers (int): Number of LSTM layers.
            output_size (int): Number of output features.
            dropout (float): Dropout rate.
            seq_length (int): Sequence length.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.device = device

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
