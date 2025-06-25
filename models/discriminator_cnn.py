import torch.nn as nn

class Discriminator_512(nn.Module):
    def __init__(self, nc= 1, nf = 8 ):
        super().__init__()
        self.block = lambda x,y : [
            nn.Conv2d(x, y, 4, 2, 1, bias=False),
            nn.BatchNorm2d(y),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False), #4
            nn.LeakyReLU(0.2, inplace=True),
            *self.block(nf, nf*2),
            *self.block(nf*2, nf*4),
            *self.block(nf*4, nf*8),
            *self.block(nf*8, nf*16),
            *self.block(nf*16, nf*32),
            # *self.block(nf*32, nf*64),
            nn.Conv2d(nf*32, 1, 4, 1, 0, bias= True), 
            nn.Sigmoid()

    )
    def forward(self, x):
        return self.main(x).view(-1, 1)


class Discriminator_502(nn.Module):
    def __init__(self, nc= 1, nf = 8 ):
        super().__init__()
        self.block = lambda x,y : [
            nn.Conv2d(x, y, 4, 2, 1, bias=False),
            nn.BatchNorm2d(y),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 3, 2, 1, bias=False), #4
            nn.LeakyReLU(0.2, inplace=True),
            *self.block(nf, nf*2),
            *self.block(nf*2, nf*4),
            *self.block(nf*4, nf*8),
            *self.block(nf*8, nf*16),
            *self.block(nf*16, nf*32),
            *self.block(nf*32, nf*64),
            nn.Conv2d(nf*64, 1, 3, 2, 0, bias= True), #4
            nn.Sigmoid()

    )
    def forward(self, x):
        return self.main(x).view(-1, 1)
