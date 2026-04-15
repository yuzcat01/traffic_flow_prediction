import os
from pathlib import Path

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
