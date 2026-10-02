"""
Training script for GAN using PyTorch.

Usage:
    python train/train_gan.py --epochs 50
    python train/train_gan.py --test-only
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.generator import Generator, Generator_cnn
from models.discriminator import Discriminator
from utils import get_device, get_transform, plot_images, plot_losses, save_image_grid, set_seed

GENERATORS = {"mlp": Generator, "cnn": Generator_cnn}


def train_gan(
    epochs=10,
    batch_size=64,
    noise_dim=100,
    lr=0.0002,
    generator_type="cnn",
    data_dir="data",
    checkpoint_dir="checkpoints",
    image_dir="images",
    num_workers=4,
    device=None
):
    """
    Train a simple GAN on MNIST dataset.
    """
    device = get_device(device)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data loader
    dataloader = DataLoader(
        datasets.MNIST(data_dir, train=True, download=True, transform=get_transform()),
        batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"), num_workers=num_workers
    )
    img_shape = tuple(dataloader.dataset[0][0].shape)

    # Models
    generator = GENERATORS[generator_type](noise_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    print("number of parameters in generator: ", sum(p.numel() for p in generator.parameters()))
    # Loss and optimizers
    adversarial_loss = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            imgs = batch[0]
            real_imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), noise_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if i == len(dataloader) - 1:
                print(f"[Epoch {epoch + 1}/{epochs}] [Batch {i + 1}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                save_image_grid(gen_imgs[:25], os.path.join(image_dir, f"epoch_{epoch + 1}.png"), nrow=5)

    # Save the generator with the settings needed to rebuild it
    checkpoint_path = os.path.join(checkpoint_dir, "generator.pth")
    torch.save({
        "state_dict": generator.state_dict(),
        "generator_type": generator_type,
        "noise_dim": noise_dim,
        "img_shape": img_shape,
    }, checkpoint_path)
    print(f"Generator saved to '{checkpoint_path}'")

    plot_losses({"Generator Loss": g_losses, "Discriminator Loss": d_losses},
                os.path.join(image_dir, "gan_loss_curve.png"), "GAN Training Losses")
    return generator


def test_gan(checkpoint_dir="checkpoints", image_dir="images", num_images=25, device=None):
    """
    Test the trained GAN by generating images.
    """
    device = get_device(device)
    os.makedirs(image_dir, exist_ok=True)
    checkpoint = torch.load(os.path.join(checkpoint_dir, "generator.pth"), map_location=device)
    noise_dim = checkpoint["noise_dim"]
    generator = GENERATORS[checkpoint["generator_type"]](noise_dim, checkpoint["img_shape"]).to(device)
    generator.load_state_dict(checkpoint["state_dict"])
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, noise_dim, device=device)
        gen_imgs = generator(z)
    output_path = os.path.join(image_dir, "generated_images.png")
    plot_images(gen_imgs, output_path)
    print(f"Generated images saved to '{output_path}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--generator", choices=sorted(GENERATORS), default="cnn")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-only", action="store_true",
                        help="skip training and generate images from the saved checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    if not args.test_only:
        train_gan(
            epochs=args.epochs,
            batch_size=args.batch_size,
            noise_dim=args.noise_dim,
            lr=args.lr,
            generator_type=args.generator,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            image_dir=args.image_dir,
            num_workers=args.num_workers,
            device=args.device,
        )
    test_gan(checkpoint_dir=args.checkpoint_dir, image_dir=args.image_dir, device=args.device)
