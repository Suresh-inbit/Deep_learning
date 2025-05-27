"""
Discriminator model for GAN (PyTorch).
"""
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator network for GAN.
    """
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        """
        Forward pass for discriminator.
        Args:
            img (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Probability of image being real.
        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
