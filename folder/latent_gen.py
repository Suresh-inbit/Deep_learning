import torch 
import torch.nn as nn
from Deep_learning.models.generator_cnn import Generator_512
from Deep_learning.models.discriminator_cnn import Discriminator_512

# from workspace.GAN_model import Discriminator
# from workspace.pin_mem import weights_init
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
netG = Generator_512().to(device)
# netD = Discriminator()
print(netG)
netG.eval()
# netG.apply()
netG.load_state_dict(torch.load('./Deep_learning/gen.pth', map_location = device))

class Latent(nn.Module):
    def __init__(self, img_shp, nf=64):
        super(Latent , self).__init__()
        self.shape = img_shp
        self.encoder = nn.Sequential(
            # Input: (3, 512, 512)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 256, 256)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 64, 64)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 32, 32)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> (512, 16, 16)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # -> (512, 8, 8)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # -> (512, 4, 4)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # -> (512, 2, 2)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=2, stride=1),             # -> (512, 1, 1)
            nn.ReLU()
        )
        self.fc = nn.Linear(512, 128)
        # self.model = nn.Sequential(
        #     nn.Linear(img_shp, img_shp//4), # image_shape -> latent size

        #     nn.ReLU(),

        #     nn.Linear(),
        #     nn.ReLU()

        # )
    def forward(self, image):
        x = self.encoder(image)  # shape: (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, 512)
        x = self.fc(x)  # -> (batch_size, 128)
        return x
    

#training parameters
epochs = 10
data_path = './Deep_learning/MvTec'
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize([0.5], [0.5]), # normalize image with mean and standard deviation.
        transforms.Resize((512, 512)),  # Reduce image size to save memory
    ])

# Dataset
dataset = ImageFolder(data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size = 8)
sample_image = next(iter(dataloader))[0].to(device)
print(sample_image.shape)

model = Latent([128,128]).to(device)
print(model(sample_image).shape)

criterion = nn.BCELoss()
optim = nn.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
for epoch in range(epochs):
    for _, image in enumerate(dataloader):
        model.zero_grad()
        image = image[0].to(device)
        latent = model(image)
        latent = latent.reshape(8,128,1,1)
        output = netG(latent)
        print(image.shape, output.shape)

        loss = criterion()

        break

