import torch
import torch.nn as nn


class STModel(nn.Module):
    def __init__(
        self,
        spatial_encoder,
        temporal_encoder,
        output_dim: int = 1,
        predict_steps: int = 1,
        head_type: str = "linear",
        pred_hidden_dim: int = 64,
        horizon_emb_dim: int = 8,
        output_dropout: float = 0.0,
        use_last_value_residual: bool = True,
    ):
        super(STModel, self).__init__()
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder
        self.output_dim = int(output_dim)
        self.predict_steps = int(predict_steps)
        self.head_type = str(head_type).strip().lower()
        self.use_last_value_residual = bool(use_last_value_residual)

        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if self.predict_steps <= 0:
            raise ValueError("predict_steps must be > 0")

        temporal_hidden = int(self.temporal_encoder.hidden_dim)
        self.output_dropout = nn.Dropout(float(max(output_dropout, 0.0)))

        if self.head_type == "linear":
            self.output_head = nn.Linear(temporal_hidden, self.predict_steps * self.output_dim)
        elif self.head_type == "horizon_mlp":
            self.horizon_embedding = nn.Embedding(self.predict_steps, int(horizon_emb_dim))
            self.output_head = nn.Sequential(
                nn.Linear(temporal_hidden + int(horizon_emb_dim), int(pred_hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(max(output_dropout, 0.0))),
                nn.Linear(int(pred_hidden_dim), self.output_dim),
            )
        else:
            raise ValueError("head_type must be one of: linear, horizon_mlp")

        if self.use_last_value_residual:
            self.residual_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, data, device):
        graph = data["graph"].to(device)
        flow_x = data["flow_x"].to(device)  # [B, N, T, D]

        if graph.dim() == 3:
            graph = graph[0]

        B, N, T, D = flow_x.shape

        spatial_outputs = []
        for t in range(T):
            x_t = flow_x[:, :, t, :]                    # [B, N, D]
            spatial_t = self.spatial_encoder(x_t, graph)  # [B, N, hidden]
            spatial_outputs.append(spatial_t)

        spatial_seq = torch.stack(spatial_outputs, dim=0).permute(1, 2, 0, 3)  # [B, N, T, C]
        temporal_out = self.temporal_encoder(spatial_seq)                       # [B, N, hidden]

        if self.head_type == "linear":
            pred = self.output_head(self.output_dropout(temporal_out))          # [B, N, H_out * D_out]
            pred = pred.view(B, N, self.predict_steps, self.output_dim)         # [B, N, H_out, D_out]
        else:
            temporal_expand = temporal_out.unsqueeze(2).expand(B, N, self.predict_steps, temporal_out.size(-1))
            horizon_ids = torch.arange(self.predict_steps, device=temporal_out.device)
            horizon_emb = self.horizon_embedding(horizon_ids).view(1, 1, self.predict_steps, -1).expand(B, N, -1, -1)
            head_input = torch.cat([self.output_dropout(temporal_expand), horizon_emb], dim=-1)
            pred = self.output_head(head_input)

        if self.use_last_value_residual:
            last_value = flow_x[:, :, -1:, :]
            if last_value.size(-1) >= self.output_dim:
                last_value = last_value[:, :, :, :self.output_dim]
            else:
                repeat_count = (self.output_dim + last_value.size(-1) - 1) // last_value.size(-1)
                last_value = last_value.repeat(1, 1, 1, repeat_count)[:, :, :, :self.output_dim]
            last_value = last_value.repeat(1, 1, self.predict_steps, 1)
            residual_scale = 2.0 * torch.sigmoid(self.residual_logit)
            pred = pred + residual_scale * last_value

        return pred
