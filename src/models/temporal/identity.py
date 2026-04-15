import torch
import torch.nn as nn


class IdentityTemporal(nn.Module):
    """
    不使用时间模块:
    输入 [B, N, T, C]
    输出 [B, N, C]
    这里直接取最后一个时间步
    """

    def __init__(self, input_dim: int):
        super(IdentityTemporal, self).__init__()
        self.hidden_dim = input_dim

    def forward(self, x):
        # x: [B, N, T, C]
        return x[:, :, -1, :]