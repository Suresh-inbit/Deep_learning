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

class Discriminator_cnn(nn.Module):
    def __init__(self, image_shape, nf= 256):
        super(Discriminator_cnn, self).__init__()
        # self.ngpu = ngpu
        nc, ndf = 1, nf
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. ``(ndf*8) x 4 x 4``

            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)