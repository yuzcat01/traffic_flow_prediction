import torch
import torch.nn as nn

from src.models.spatial.graph_ops import symmetric_normalize_adjacency


class GCNSpatial(nn.Module):
    def __init__(self, in_c: int, hidden_dim: int, dropout: float = 0.0):
        super(GCNSpatial, self).__init__()
        self.linear = nn.Linear(in_c, hidden_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(max(dropout, 0.0)))
        self.hidden_dim = hidden_dim

    def forward(self, x, graph):
        graph = self.process_graph(graph)
        out = self.linear(x)
        out = torch.matmul(graph, out)
        out = self.act(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out

    @staticmethod
    def process_graph(graph):
        return symmetric_normalize_adjacency(graph, add_self_loop=True)
