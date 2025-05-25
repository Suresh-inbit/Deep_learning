"""
Training script for GAN using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append("..")
from models.generator import Generator
from models.discriminator import Discriminator
import torchvision
from torchvision4ad.datasets import MVTecAD


def train_gan(
    epochs=10,
    batch_size=64,
    noise_dim=100,
    lr=0.0002,
    device=None
):
    """
    Train a simple GAN on MNIST dataset.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("images", exist_ok=True)

    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize([0.5], [0.5]),
        # transforms.Resize((256, 256)),  # Reduce image size to save memory
    ])
    dataloader = DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )
    # print("size of images: ", dataloader.dataset[0][0].shape)
    img_shape = dataloader.dataset[0][0].shape
    # batch_size = 2  # Reduce batch size to save memory
    # dataset = MVTecAD("MVTec", 'grid', train=True, transform=transform, download=True)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print("size of images: ", dataset[1][0].shape)
    # img_shape = dataset[0][0].shape
    # plt.imshow(dataset[0][0].numpy().squeeze(), cmap='gray')
    # plt.axis('off')
    # plt.show()
    # print("label of images: ", [i[1] for i in dataset])
    # print("size of images: ", img_shape)
    # exit(0)
    # Models
    generator = Generator(noise_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    print("number of parameters in generator: ", sum(p.numel() for p in generator.parameters()))
    # Loss and optimizers
    adversarial_loss = nn.BCELoss()
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
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                save_image(gen_imgs.data[:25], f"images/{epoch}_{i}.png", nrow=5)

    # Plot losses after training
    plt.figure(figsize=(10,5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.savefig('images/loss_curve.png')
    plt.close()

def save_image(tensor, filename, nrow=5):
    """
    Save a grid of images.
    """
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True)
    # ndarr = grid.mul(127.5).add(127.5).clamp(0,255).byte().cpu().numpy()
    plt.imsave(filename, np.transpose(grid.cpu().numpy(), (1,2,0)))

if __name__ == "__main__":
    train_gan(epochs=50)
