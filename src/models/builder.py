from src.models.spatial.gcn import GCNSpatial
from src.models.spatial.chebnet import ChebNetSpatial
from src.models.spatial.gat import GATSpatial
from src.models.temporal.gru import GRUTemporal
from src.models.temporal.identity import IdentityTemporal
from src.models.temporal.tcn import TCNTemporal
from src.models.st_model import STModel


def build_spatial_encoder(model_cfg):
    spatial_cfg = model_cfg["spatial"]
    spatial_type = spatial_cfg["type"]
    in_c = model_cfg["input"]["input_dim"]
    hidden_dim = spatial_cfg["hidden_dim"]
    regularization_cfg = model_cfg.get("regularization", {})
    dropout = float(regularization_cfg.get("dropout", 0.0))

    if spatial_type == "gcn":
        return GCNSpatial(in_c=in_c, hidden_dim=hidden_dim, dropout=dropout)
    elif spatial_type == "chebnet":
        return ChebNetSpatial(
            in_c=in_c,
            hidden_dim=hidden_dim,
            K=spatial_cfg.get("cheb_k", 3),
            dropout=dropout,
        )
    elif spatial_type == "gat":
        return GATSpatial(
            in_c=in_c,
            hidden_dim=hidden_dim,
            heads=spatial_cfg.get("heads", 1),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported spatial_type: {spatial_type}")


def build_temporal_encoder(model_cfg):
    temporal_cfg = model_cfg["temporal"]
    temporal_type = temporal_cfg["type"]
    spatial_hidden = model_cfg["spatial"]["hidden_dim"]
    regularization_cfg = model_cfg.get("regularization", {})
    dropout = float(regularization_cfg.get("dropout", 0.0))

    if temporal_type == "gru":
        return GRUTemporal(
            input_dim=spatial_hidden,
            hidden_dim=temporal_cfg["hidden_dim"],
            num_layers=int(temporal_cfg.get("num_layers", 1)),
            dropout=dropout,
        )
    elif temporal_type == "tcn":
        return TCNTemporal(
            input_dim=spatial_hidden,
            hidden_dim=temporal_cfg["hidden_dim"],
            num_layers=int(temporal_cfg.get("num_layers", 2)),
            kernel_size=int(temporal_cfg.get("kernel_size", 3)),
            dropout=dropout,
        )
    elif temporal_type == "none":
        return IdentityTemporal(input_dim=spatial_hidden)
    else:
        raise ValueError(f"Unsupported temporal_type: {temporal_type}")


def build_model(cfg):
    model_cfg = cfg["model"]

    spatial_encoder = build_spatial_encoder(model_cfg)
    temporal_encoder = build_temporal_encoder(model_cfg)
    output_cfg = model_cfg.get("output", {})
    output_dim = int(output_cfg.get("output_dim", 1))
    predict_steps = int(output_cfg.get("predict_steps", 1))
    head_type = str(output_cfg.get("head_type", "linear")).strip().lower()
    pred_hidden_dim = int(output_cfg.get("pred_hidden_dim", 64))
    horizon_emb_dim = int(output_cfg.get("horizon_emb_dim", 8))
    output_dropout = float(output_cfg.get("dropout", model_cfg.get("regularization", {}).get("dropout", 0.0)))
    use_last_value_residual = bool(output_cfg.get("use_last_value_residual", True))

    return STModel(
        spatial_encoder=spatial_encoder,
        temporal_encoder=temporal_encoder,
        output_dim=output_dim,
        predict_steps=predict_steps,
        head_type=head_type,
        pred_hidden_dim=pred_hidden_dim,
        horizon_emb_dim=horizon_emb_dim,
        output_dropout=output_dropout,
        use_last_value_residual=use_last_value_residual,
    )
