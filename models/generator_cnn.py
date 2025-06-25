import torch
import torch.nn as nn

  
class Generator_512(nn.Module):
    def __init__(self, latent_dim=128, nf=52):
        super().__init__()
        self.block = lambda in_channels, out_channels: [
            nn.Upsample(scale_factor=2, mode='bilinear'), 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels),
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
            # *self.block(nf, nf//2), 
            nn.ConvTranspose2d(nf, 1, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
    
class Generator_502(nn.Module):
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
