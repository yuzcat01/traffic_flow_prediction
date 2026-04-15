import torch
import torch.nn as nn


class ChebConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, K: int):
        super(ChebConv, self).__init__()
        self.K = K
        self.linear = nn.Linear(in_c * K, out_c)

    def forward(self, x, graph):
        laplacian = self.get_laplacian(graph)
        cheb_polynomials = self.cheb_polynomial(laplacian, self.K)

        out_list = []
        for k in range(self.K):
            T_k = cheb_polynomials[k]
            out_k = torch.matmul(T_k, x)
            out_list.append(out_k)

        out = torch.cat(out_list, dim=-1)
        out = self.linear(out)
        return out

    @staticmethod
    def get_laplacian(graph):
        N = graph.size(0)
        I = torch.eye(N, dtype=torch.float32, device=graph.device)

        degree = torch.sum(graph, dim=1)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0.0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)

        A_norm = torch.mm(torch.mm(D_inv_sqrt, graph), D_inv_sqrt)
        L = I - A_norm
        return L

    @staticmethod
    def cheb_polynomial(laplacian, K):
        N = laplacian.size(0)
        I = torch.eye(N, dtype=torch.float32, device=laplacian.device)

        multi_order_laplacian = [I]
        if K == 1:
            return multi_order_laplacian

        multi_order_laplacian.append(laplacian)

        for k in range(2, K):
            T_k = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]
            multi_order_laplacian.append(T_k)

        return multi_order_laplacian


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
