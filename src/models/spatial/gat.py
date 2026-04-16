import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spatial.graph_ops import build_binary_attention_mask


class GraphAttentionLayer(nn.Module):
    """
    简化版图注意力层
    输入:
        inputs: [B, N, C]
        graph:  [N, N]
    输出:
        out:    [B, N, D]
    """

    def __init__(self, in_c: int, out_c: int, dropout: float = 0.0, negative_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.negative_slope = float(negative_slope)

        self.proj = nn.Linear(in_c, out_c, bias=False)
        self.attn_src = nn.Parameter(torch.empty(out_c))
        self.attn_dst = nn.Parameter(torch.empty(out_c))
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.attn_dropout = nn.Dropout(float(max(dropout, 0.0)))
        self.residual_proj = nn.Linear(in_c, out_c, bias=False) if in_c != out_c else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))
        nn.init.zeros_(self.bias)

    def forward(self, inputs, graph):
        """
        inputs: [B, N, C]
        graph: [N, N]
        """
        h = self.proj(inputs)  # [B, N, D]
        src_logits = torch.matmul(h, self.attn_src)  # [B, N]
        dst_logits = torch.matmul(h, self.attn_dst)  # [B, N]

        scores = src_logits.unsqueeze(2) + dst_logits.unsqueeze(1)  # [B, N, N]
        scores = F.leaky_relu(scores, negative_slope=self.negative_slope)

        mask = graph.to(dtype=torch.bool, device=inputs.device).unsqueeze(0)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attention = F.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)

        out = torch.bmm(attention, h)
        out = out + self.residual_proj(inputs) + self.bias.view(1, 1, -1)
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
        self.heads = max(1, int(heads))
        self.dropout_rate = float(max(dropout, 0.0))

        self.attention_module = nn.ModuleList(
            [GraphAttentionLayer(in_c, hidden_dim, dropout=self.dropout_rate) for _ in range(self.heads)]
        )

        self.out_att = GraphAttentionLayer(hidden_dim * self.heads, hidden_dim, dropout=self.dropout_rate)
        self.act = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, graph):
        """
        x: [B, N, C]
        graph: [N, N]
        """
        graph = build_binary_attention_mask(graph, symmetric=True, add_self_loop=True)

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
