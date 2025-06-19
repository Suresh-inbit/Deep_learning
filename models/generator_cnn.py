import torch
import torch.nn as nn

  
class Generator_512(nn.Module):
    def __init__(self, latent_dim=128, nf=52):
        super().__init__()
        self.block = lambda x, y: [
            nn.ConvTranspose2d(x, y, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(y),
            nn.ReLU(True)
        ]
        
        self.model = nn.Sequential(
            # Initial projection: latent_dim → 4x4 feature map
            nn.ConvTranspose2d(latent_dim, nf*64, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf*64),
            nn.ReLU(True),
            
            # 7 upsampling blocks (4x4 → 512x512)
            *self.block(nf*64, nf*32),   # 8x8
            *self.block(nf*32, nf*16),    # 16x16
            *self.block(nf*16, nf*8),    # 32x32
            *self.block(nf*8, nf*4),      # 64x64
            *self.block(nf*4, nf*2),     # 128x128
            *self.block(nf*2, nf),  # 256x256
            nn.ConvTranspose2d(nf, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Generator_cnn(nn.Module):
    """
    CNN Generator for GAN that generates 512x512 images from a 128-dimensional input.
    """
    def __init__(self, input_dim=128, output_shape=(1, 512, 512), ngf=64):
        super(Generator_cnn, self).__init__()
        self.output_shape = output_shape
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size: (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # state size: (ngf//2) x 128 x 128
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            # state size: (ngf//4) x 256 x 256
            nn.ConvTranspose2d(ngf // 4, output_shape[0], 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (output_shape[0]) x 512 x 512
        )

    def forward(self, z):
        """
        Forward pass for CNN generator.
        Args:
            z (torch.Tensor): Input noise vector of shape (batch_size, input_dim, 1, 1).
        Returns:
            torch.Tensor: Generated image of shape (batch_size, output_shape[0], 512, 512).
        """
        z = z.view(z.size(0), -1, 1, 1)  # Reshape input to (batch_size, input_dim, 1, 1)
        img = self.model(z)
        return img

# class Generator_cnn(nn.Module):
#     """
#     CNN Generator for GAN that generates 256x256 images from a 128-dimensional input.
#     """
#     def __init__(self, input_dim, output_shape, ngf):
#         super(Generator_cnn, self).__init__()
#         self.output_shape = output_shape
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(input_dim, ngf * 16, 5, 3, 0, bias=False),
#             nn.BatchNorm2d(ngf * 16),
#             nn.ReLU(True),
#             # state size. ``(ngf*16) x 4 x 4``
#             nn.ConvTranspose2d(ngf * 16, ngf * 8, 5, 3, 1, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. ``(ngf*8) x 8 x 8``
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. ``(ngf*4) x 16 x 16``
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. ``(ngf*2) x 32 x 32``
#             nn.ConvTranspose2d(ngf * 2, ngf, 5, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. ``(ngf) x 64 x 64``
#             nn.ConvTranspose2d(ngf , ngf, 5, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),           
#             nn.ConvTranspose2d(ngf, output_shape[0], 4, 2, 0, bias=False),
#             nn.Tanh()
#             # state size. ``(output_shape[0]) x 128 x 128``
#         )

#     def forward(self, z):
#         """
#         Forward pass for CNN generator.
#         Args:
#             z (torch.Tensor): Input noise vector of shape (batch_size, input_dim, 1, 1).
#         Returns:
#             torch.Tensor: Generated image of shape (batch_size, output_shape[0], 256, 256).
#         """
#         z = z.view(z.size(0), -1, 1, 1)  # Reshape input to (batch_size, input_dim, 1, 1)
#         img = self.model(z)
#         return img
