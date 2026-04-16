import torch
import torch.nn as nn

from src.models.spatial.graph_ops import scaled_laplacian


class ChebConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, K: int):
        super(ChebConv, self).__init__()
        self.K = max(1, int(K))
        self.linear = nn.Linear(in_c * self.K, out_c)

    def forward(self, x, graph):
        laplacian = self.get_scaled_laplacian(graph)
        out = self.cheb_feature_stack(x, laplacian, self.K)
        out = self.linear(out)
        return out

    @staticmethod
    def get_scaled_laplacian(graph):
        return scaled_laplacian(graph, add_self_loop=True)

    @staticmethod
    def cheb_feature_stack(x, scaled_laplacian, K):
        terms = [x]
        if K == 1:
            return torch.cat(terms, dim=-1)

        t_k_minus_two = x
        t_k_minus_one = torch.matmul(scaled_laplacian, x)
        terms.append(t_k_minus_one)

        for _ in range(2, K):
            t_k = 2.0 * torch.matmul(scaled_laplacian, t_k_minus_one) - t_k_minus_two
            terms.append(t_k)
            t_k_minus_two, t_k_minus_one = t_k_minus_one, t_k

        return torch.cat(terms, dim=-1)


class ChebNetSpatial(nn.Module):
    def __init__(self, in_c: int, hidden_dim: int, K: int = 3, dropout: float = 0.0):
        super(ChebNetSpatial, self).__init__()
        self.cheb = ChebConv(in_c, hidden_dim, K)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(max(dropout, 0.0)))
        self.hidden_dim = hidden_dim

    def forward(self, x, graph):
        out = self.cheb(x, graph)
        out = self.act(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out
