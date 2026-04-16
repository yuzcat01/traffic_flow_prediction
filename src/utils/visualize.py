import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from PIL import Image
except Exception:
    Image = None


def _sanitize_png_for_qt(save_path: str):
    if Image is None:
        return

    path = Path(save_path)
    if path.suffix.lower() != ".png" or not path.exists():
        return

    try:
        with Image.open(path) as image:
            image.convert("RGB").save(path, format="PNG")
    except Exception:
        return


def _save_figure(fig, save_path: str):
    fig.savefig(
        save_path,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    _sanitize_png_for_qt(save_path)
    plt.close(fig)


def plot_prediction_vs_target(target, prediction, save_path, title="Prediction vs Target"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(target, label="Target")
    ax.plot(prediction, label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Traffic Flow")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_prediction_overview(
    targets,
    predictions,
    save_path,
    title="Prediction Overview",
    figure_points=300,
    focus_node_id=0,
    focus_horizon_step=0,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if targets.shape != predictions.shape or targets.ndim != 4:
        raise ValueError("targets and predictions must have shape [samples, nodes, horizons, dims]")

    sample_count, num_nodes, predict_steps, _ = targets.shape
    focus_node_id = min(max(int(focus_node_id), 0), num_nodes - 1)
    focus_horizon_step = min(max(int(focus_horizon_step), 0), predict_steps - 1)
    figure_points = max(10, min(int(figure_points), sample_count))

    abs_error = np.abs(predictions - targets)
    node_mae = abs_error.mean(axis=(0, 2, 3))
    ranked_nodes = np.argsort(node_mae)
    best_node = int(ranked_nodes[0])
    median_node = int(ranked_nodes[len(ranked_nodes) // 2])
    worst_node = int(ranked_nodes[-1])

    horizon_mae = abs_error.mean(axis=(0, 1, 3))
    horizon_rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=(0, 1, 3)))

    node_horizon_mae = abs_error.mean(axis=(0, 3)).T  # [H, N]
    max_heatmap_nodes = min(64, num_nodes)
    heatmap_nodes = np.argsort(-node_mae)[:max_heatmap_nodes]
    heatmap_data = node_horizon_mae[:, heatmap_nodes]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=14)

    # 1) Focused node + horizon trend for continuity with old figure
    ax = axes[0, 0]
    target_curve = targets[:figure_points, focus_node_id, focus_horizon_step, 0]
    pred_curve = predictions[:figure_points, focus_node_id, focus_horizon_step, 0]
    ax.plot(target_curve, label="Target", color="#059669", linewidth=2)
    ax.plot(pred_curve, label="Prediction", color="#f59e0b", linewidth=2)
    ax.set_title(f"Focus Node {focus_node_id} | Horizon H{focus_horizon_step + 1}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Traffic Flow")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    # 2) Representative nodes on the selected horizon
    ax = axes[0, 1]
    for node_id, name, color in [
        (best_node, "Best node", "#2563eb"),
        (median_node, "Median node", "#7c3aed"),
        (worst_node, "Worst node", "#dc2626"),
    ]:
        ax.plot(
            targets[:figure_points, node_id, focus_horizon_step, 0],
            label=f"{name} target (node {node_id})",
            linestyle="--",
            color=color,
            alpha=0.85,
        )
        ax.plot(
            predictions[:figure_points, node_id, focus_horizon_step, 0],
            label=f"{name} pred",
            color=color,
            linewidth=2,
        )
    ax.set_title(f"Representative Nodes | Horizon H{focus_horizon_step + 1}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Traffic Flow")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # 3) Full-horizon metrics trend
    ax = axes[1, 0]
    horizon_x = np.arange(1, predict_steps + 1)
    ax.plot(horizon_x, horizon_mae, marker="o", color="#2563eb", label="MAE")
    ax.plot(horizon_x, horizon_rmse, marker="s", color="#f59e0b", label="RMSE")
    ax.set_title("Full Horizon Metrics")
    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(horizon_x)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    # 4) Error heatmap across horizons and high-error nodes
    ax = axes[1, 1]
    image = ax.imshow(heatmap_data, aspect="auto", origin="lower", cmap="YlOrRd")
    ax.set_title(f"Node-Horizon MAE Heatmap (Top {max_heatmap_nodes} nodes by error)")
    ax.set_xlabel("Node Rank by Error")
    ax.set_ylabel("Horizon Step")
    ax.set_yticks(np.arange(predict_steps))
    ax.set_yticklabels([f"H{i}" for i in horizon_x])
    ax.set_xticks(np.arange(len(heatmap_nodes)))
    ax.set_xticklabels([str(int(node_id)) for node_id in heatmap_nodes], rotation=90, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="MAE")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, save_path)


def plot_loss_curve(train_losses, val_losses, save_path, title="Loss Curve"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss")
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(val_losses, label="Val Loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path)
