import os
import matplotlib.pyplot as plt


def plot_prediction_vs_target(target, prediction, save_path, title="Prediction vs Target"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(target, label="Target")
    plt.plot(prediction, label="Prediction")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_curve(train_losses, val_losses, save_path, title="Loss Curve"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    if val_losses is not None and len(val_losses) > 0:
        plt.plot(val_losses, label="Val Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()