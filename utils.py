"""
Shared helpers for the training scripts.
"""
import math
import random

import matplotlib
matplotlib.use("Agg")  # Save figures to disk; never block on a display window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms


def get_transform():
    """
    Transform used for every dataset: 1-channel tensor normalized to [-1, 1].
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize([0.5], [0.5]),
    ])


def get_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_grid(tensor, filename, nrow=5):
    """
    Save a batch of images as a single grid image.
    """
    grid = torchvision.utils.make_grid(tensor.detach().cpu(), nrow=nrow, normalize=True)
    plt.imsave(filename, np.transpose(grid.numpy(), (1, 2, 0)))


def plot_images(images, filename, titles=None):
    """
    Save a square grid of images, optionally with a title above each one.
    """
    n = len(images)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    for i in range(n):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i].detach().cpu().numpy().squeeze(), cmap="gray")
        if titles is not None:
            plt.title(titles[i], fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_losses(losses, filename, title):
    """
    Save a loss curve. `losses` maps a label to a list of per-iteration values.
    """
    plt.figure(figsize=(10, 5))
    for label, values in losses.items():
        plt.plot(values, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
