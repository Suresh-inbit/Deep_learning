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
    """
    DCGAN-style generator: projects the noise vector to a small feature map,
    then upsamples it 4x with transposed convolutions.
    Image height and width must be divisible by 4 (e.g. 28x28 -> 7x7 start).
    """
    def __init__(self, latent_dim=100, image_shape=(1, 28, 28)):
        super(Generator_cnn, self).__init__()
        channels, height, width = image_shape
        if height % 4 or width % 4:
            raise ValueError(f"image height and width must be divisible by 4, got {image_shape}")
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.init_shape = (256, height // 4, width // 4)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, int(torch.prod(torch.tensor(self.init_shape)))),
            nn.BatchNorm1d(int(torch.prod(torch.tensor(self.init_shape)))),
            nn.ReLU(True),
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # x2
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # x4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Noise of shape (batch_size, latent_dim).
        Returns:
            torch.Tensor: Images of shape (batch_size, *image_shape).
        """
        x = self.project(z).view(z.size(0), *self.init_shape)
        return self.model(x)