"""Transformer captures cross-stock relationships & market trends.
I plan to combine this with the LSTM for ensemble predictions.
"""
    
import torch
import torch.nn as nn

class TransformerStockPredictor(nn.Module):
    def __init__(self, input_size, num_heads, hidden_dim, num_layers):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x[:, -1, :])  # will predict the next-day price
