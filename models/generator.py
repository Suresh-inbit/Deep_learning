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
    def __init__(self, latent_dim = 128, image_shape=(1, 28, 28), nz=128, ngf=28, nc=1):
        super(Generator_cnn, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        if nz is None:
            nz = latent_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
