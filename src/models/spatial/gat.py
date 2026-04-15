import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    简化版图注意力层
    输入:
        inputs: [B, N, C]
        graph:  [N, N]
    输出:
        out:    [B, N, D]
    """

    def __init__(self, in_c: int, out_c: int):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.W = nn.Linear(in_c, out_c, bias=False)
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.b)

    def forward(self, inputs, graph):
        """
        inputs: [B, N, C]
        graph: [N, N]
        """
        h = self.W(inputs)   # [B, N, D]

        # 点积注意力分数
        scores = torch.bmm(h, h.transpose(1, 2))   # [B, N, N]

        # 只保留有边的位置
        graph_mask = graph.unsqueeze(0).expand_as(scores)   # [B, N, N]
        scores = scores.masked_fill(graph_mask == 0, -1e16)

        attention = F.softmax(scores, dim=2)   # [B, N, N]
        out = torch.bmm(attention, h) + self.b  # [B, N, D]

        return out


class GATSpatial(nn.Module):
    """
    统一框架下的 GAT 空间编码器
    输入:
        x: [B, N, C]
        graph: [N, N]
    输出:
        out: [B, N, hidden_dim]
    """

    def __init__(self, in_c: int, hidden_dim: int, heads: int = 1, dropout: float = 0.0):
        super(GATSpatial, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.attention_module = nn.ModuleList(
            [GraphAttentionLayer(in_c, hidden_dim) for _ in range(heads)]
        )

        self.out_att = GraphAttentionLayer(hidden_dim * heads, hidden_dim)
        self.act = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(max(dropout, 0.0)))

    def forward(self, x, graph):
        """
        x: [B, N, C]
        graph: [N, N]
        """
        # 加自环
        N = graph.size(0)
        device = graph.device
        graph = graph + torch.eye(N, device=device)
        graph = (graph > 0).float()

        outputs = torch.cat(
            [attn(x, graph) for attn in self.attention_module],
            dim=-1
        )   # [B, N, hidden_dim * heads]

        outputs = self.act(outputs)
        outputs = self.out_att(outputs, graph)   # [B, N, hidden_dim]
        outputs = self.act(outputs)
        outputs = self.norm(outputs)
        outputs = self.dropout(outputs)

        return outputs
