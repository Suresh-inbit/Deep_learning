"""
Generator model for GAN (PyTorch).
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator network for GAN.
    """
    def __init__(self, noise_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass for generator.
        Args:
            z (torch.Tensor): Input noise vector.
        Returns:
            torch.Tensor: Generated image.
        """
        img = self.model(z)
        # print(f"img shape: {img.shape}")
        img = img.view(z.size(0), *self.img_shape)
        return img
class Generator_cnn(nn.Module):
    def __init__(self, latent_dim = 128, image_shape=(1, 28, 28)):
        super(Generator_cnn, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.model = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim * 4, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(latent_dim * 4, latent_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(latent_dim, image_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), *self.image_shape)
        return img