import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_shape, nf = 64):
        super().__init__()
        self.image_shape = image_shape
        self.model=nn.Sequential(
            nn.Conv2d(image_shape[0], nf* 4, kernel_size=4, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            
            nn.Conv2d(nf*4, nf*2, kernel_size=4, stride=2, padding='valid'),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(nf*2, nf, kernel_size=4, stride=2, padding='valid'),
            nn.MaxPool2d(2),
            nn.Sigmoid(),

            # nn.Flatten(),
            # nn.Linear(nf* 3* 3, 1),
            # nn.Softmax()
            
        )
    def forward(self, input):
        return self.model(input)
import torch
import torch.nn as nn

class Discriminator_512(nn.Module):
    def __init__(self, nc=1, nf=12):
        super().__init__()
        # Input: (nc) x 512 x 512
        self.main = nn.Sequential(
            # (nc) x 512 x 512 → (nf) x 256 x 256
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (nf) x 256 x 256 → (nf*2) x 128 x 128
            nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (nf*2) x 128 x 128 → (nf*4) x 64 x 64
            nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # (nf*4) x 64 x 64 → (nf*8) x 32 x 32
            nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),

            # (nf*8) x 32 x 32 → (nf*16) x 16 x 16
            nn.Conv2d(nf*8, nf*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*16),
            nn.LeakyReLU(0.2, inplace=True),

            # (nf*16) x 16 x 16 → (nf*32) x 8 x 8
            nn.Conv2d(nf*16, nf*32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*32),
            nn.LeakyReLU(0.2, inplace=True),

            # (nf*32) x 8 x 8 → (nf*64) x 4 x 4
            nn.Conv2d(nf*32, nf*64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*64),
            nn.LeakyReLU(0.2, inplace=True),

            # (nf*64) x 4 x 4 → 1 x 1 x 1
            nn.Conv2d(nf*64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # For vanilla GAN; remove for WGAN
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


class Discriminator_cnn2(nn.Module):
    """
    GAN Discriminator for 512x512 images.
    """
    def __init__(self, input_channels=1, ndf=64):
        super(Discriminator_cnn2, self).__init__()
        self.model = nn.Sequential(
            # Input: (input_channels) x 512 x 512
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (ndf) x 256 x 256

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (ndf*2) x 128 x 128

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (ndf*4) x 64 x 64

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (ndf*8) x 32 x 32

            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (ndf*16) x 16 x 16

            nn.Conv2d(ndf * 16, ndf * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (ndf*32) x 8 x 8

            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


class Discriminator_cnn(nn.Module):
    """
    CNN Discriminator for GAN to classify 512x512 images as real or fake.
    """
    def __init__(self, image_shape=(1, 512, 512), nf=64):
        super(Discriminator_cnn, self).__init__()
        nc = 1  # Number of channels in the input image
        ndf = nf  # Number of filters in the first layer
        self.main = nn.Sequential(
            # Input: (nc) x 512 x 512
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf) x 256 x 256
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 128 x 128
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 64 x 64
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 32 x 32
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*16) x 16 x 16
            nn.Conv2d(ndf * 16, ndf * 20, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 20),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*32) x 8 x 8
            nn.Conv2d(ndf * 20, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*64) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Outputs a scalar probability
            # State: 1 x 1 x 1
        )

    def forward(self, input):
        """
        Forward pass for the discriminator.
        Args:
            input (torch.Tensor): Input image tensor of shape (batch_size, nc, 512, 512).
        Returns:
            torch.Tensor: Classification result (real/fake) of shape (batch_size, 1, 1, 1).
        """
        return self.main(input)  # Flatten to (batch_size, 1)

# class Discriminator_cnn(nn.Module):
#     def __init__(self, image_shape, nf= 256):
#         super(Discriminator_cnn, self).__init__()
#         # self.ngpu = ngpu
#         nc, ndf = 1, nf
#         self.main = nn.Sequential(
#             # input is ``(nc) x 64 x 64``
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf) x 32 x 32``
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*2) x 16 x 16``
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*4) x 8 x 8``
#             nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
#             # nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),

#             # state size. ``(ndf*8) x 4 x 4``

#             nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),


#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input)