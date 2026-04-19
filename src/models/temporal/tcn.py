import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size <= 1:
            raise ValueError("TCN kernel_size must be > 1")

        self.left_padding = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(max(dropout, 0.0)))
        self.residual_proj = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Conv1d(input_dim, output_dim, kernel_size=1)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = F.pad(x, (self.left_padding, 0))
        out = self.conv(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + residual
        out = out.transpose(1, 2)
        out = self.norm(out)
        return out.transpose(1, 2)


class TCNTemporal(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.kernel_size = int(kernel_size)
        self.dropout_rate = float(max(dropout, 0.0))

        layers = []
        in_dim = int(input_dim)
        for layer_idx in range(self.num_layers):
            dilation = 2 ** layer_idx
            layers.append(
                TemporalConvBlock(
                    input_dim=in_dim,
                    output_dim=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout_rate,
                )
            )
            in_dim = self.hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, C]
        batch_size, num_nodes, seq_len, channels = x.shape
        x = x.contiguous().view(batch_size * num_nodes, seq_len, channels).transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        out = out.view(batch_size, num_nodes, self.hidden_dim)
        return out
