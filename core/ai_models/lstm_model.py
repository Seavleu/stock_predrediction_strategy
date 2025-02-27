import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        
        self.batch_norm = nn.BatchNorm1d(input_size)  # normalize across input features
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_length, num_features = x.shape  # x shape: (batch_size, seq_length, input_size)

        x = x.view(batch_size * seq_length, num_features)  # flatten batch for batch norm
        x = self.batch_norm(x)  # Normalize inputs
        x = x.view(batch_size, seq_length, num_features)  # reshape back to original LSTM format

        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Predict next step