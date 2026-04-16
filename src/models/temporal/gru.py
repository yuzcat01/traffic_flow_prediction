import torch
import torch.nn as nn


class GRUTemporal(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super(GRUTemporal, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, int(num_layers))
        self.dropout_rate = float(max(dropout, 0.0))
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # x: [B, N, T, C]
        B, N, T, C = x.shape
        x = x.contiguous().view(B * N, T, C)
        self.gru.flatten_parameters()
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        last_hidden = last_hidden.view(B, N, self.hidden_dim)
        return last_hidden
