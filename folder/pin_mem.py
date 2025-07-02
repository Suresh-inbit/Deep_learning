import torch
from torchvision import transforms
import torch.nn as nn
from torchvision4ad.datasets import MVTecAD
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import random
import matplotlib.pyplot as plt
import os, math
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
manualSeed = 999

#manualSeed = random.randint(1, 10000) # use if you want new results
print("Here we go again....")
random.seed(manualSeed)
torch.manual_seed(manualSeed)


class CustomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Extract the angle from the filename
        # filename = img.imgs
        # print(filename)
        # angle = float(os.path.splitext(os.path.basename(filename))[0].split('_')[-1])

        # Calculate the cropping region
        width, height = img.size
        crop_width, crop_height = self.size

        # Calculate the maximum possible crop size to avoid black borders
        # angle_rad = math.radians(angle)
        # cos_angle = abs(math.cos(angle_rad))
        # sin_angle = abs(math.sin(angle_rad))

        # Calculate the dimensions of the largest possible rectangle that fits within the rotated image
        # max_crop_width = int(width * cos_angle + height * sin_angle)
        # max_crop_height = int(width * sin_angle + height * cos_angle)

        # Ensure the crop size does not exceed the image dimensions
        # crop_width = min(crop_width, max_crop_width)
        # crop_height = min(crop_height, max_crop_height)

        # Calculate the top-left corner of the crop
  
        top = torch.randint(0, height - crop_height -128, (1,)).item()
        left = torch.randint(0, width - crop_width -128, (1,)).item()
        # print(len(img.getdata()))
        while img.getdata()[top*512+left]==0 or img.getdata()[top*512+left+128]==0 :
            top = torch.randint(0, height - crop_height + 1, (1,)).item()
            left = torch.randint(0, width - crop_width + 1, (1,)).item()
            # print("failed once")


        # print(f"Cropping image at: top={top}, left={left}, width={crop_width}, height={crop_height}")
        return transforms.functional.crop(img, top, left, crop_height, crop_width)


# class CustomRandomCrop:
#     def __init__(self, size, limit):
#         self.size = size
#         self.limit = limit

#     def __call__(self, img):
#         # Custom random crop logic
#         width, height = img.size
#         crop_width, crop_height = self.size

#         if width < crop_width or height < crop_height:
#             raise ValueError("Crop size must be smaller than image size")

#         top = torch.randint(0, height - crop_height + 1, (1,)).item()
#         left = torch.randint(0, width - crop_width + 1, (1,)).item()

#         print(f"Cropping image at: top={top}, left={left}, width={crop_width}, height={crop_height}")
#         return transforms.functional.crop(img, top, left, crop_height, crop_width)

transform = transforms.Compose([
    transforms.Resize((512,512)),  # Resize the image to 512x512
    transforms.Grayscale(), #for running on IITM server
    transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BILINEAR),  
    CustomCrop((128, 128)),  # Randomly crop a 128x128 patch 
    # transforms.RandomCrop((128,128)),
    transforms.ToTensor()  # Convert the image to a tensor
])

# mvtec_ad_train = MVTecAD('mvtec_ad', 'grid', train=True, transform=None, download=False) #make it true if new system
# exit()
# Custom dataset to ensure each object category has 10,000 patches
## DOwnload dataset
# from torchvision.datasets.utils import download_and_extract_archive
# # tile_url ="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz"
# # download_and_extract_archive(tile_url, "mvtec_ad", 'tile.tar.tz')

# import ssl
# import urllib.request

# context = ssl._create_unverified_context()
# urllib.request.urlopen(tile_url, context=context)

class AugmentedDataset(Dataset):
    def __init__(self, dataset, num_patches_per_category=10048):
        self.dataset = dataset
        self.num_patches_per_category = num_patches_per_category
        # Calculate the number of categories
        self.num_categories = len(dataset.classes)
        self.indices = {category: [] for category in range(self.num_categories)}
        # Store indices for each category
        for idx in range(len(dataset)):

            _, label = dataset[idx]

            self.indices[label].append(idx)

    def __len__(self):

        return self.num_patches_per_category * self.num_categories

    def __getitem__(self, idx):

        # Determine the category and local index within that category

        category = idx // self.num_patches_per_category

        local_idx = random.choice(self.indices[category])

        # Get the image and label

        img, label = self.dataset[local_idx]

        return img, label

# Assume that 'mvtec_ad_path' is the path to your MVTec AD dataset

dataset = ImageFolder(root='/home/cudaq/files/workspace/mvtec_ad/grid/train', transform=transform)

# Create the augmented dataset
augmented_dataset = AugmentedDataset(dataset)
#augmented_dataset = AugmentedDataset(mvtec_ad_train)
# Create a DataLoader
print("len of dataset :", len(augmented_dataset))
batch_size=64
workers = 20
ngpu=1
train_dataloader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory= True)
device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# image = next(iter(train_dataloader))
# loaded= list(enumerate(train_dataloader))[4]
# print(loaded[1][0].shape)
# plt.figure
# figure , ax= plt.subplots(16,8)
# for row in range(8):
#     for column in range(8):
#         ax[row, column].imshow(loaded[1][0][row+(row)*column][0])
# for row in range(8,16):
#     for column in range(8):
#         ax[row, column].imshow(loaded[1][0][row-8+(row-8)*column][0])
# plt.axis('off')
# # plt.imshow(vutils.make_grid(image[0].squeeze(), normalize=True))
# plt.savefig("imagew.png",bbox_inches='tight')
# exit()

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# Number of workers for dataloader
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 1 #1 for grayscale images; else 3
# Size of z latent vector (i.e. size of generator input)
nz = 64
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 200
# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
seed_size=int(image_size/16)
seed=int(ngf*8*seed_size*seed_size)
class Generator(nn.Module): #modify this; first layer is linear , and then reshape (check if batchnorm and relu were used after this layer)

    def __init__(self, ngpu):

        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a linear layer
            nn.Linear(nz,seed,bias=True),
            nn.Unflatten(1,(ngf*8,seed_size,seed_size)),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # nn.ConvTranspose2d( ngf*8, ngf * 8, 5, 1, 1, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf,nc,5,2,2,1,bias=True),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``

        )

    def forward(self, input):
        return self.main(input)

# Create the generator

netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired

if (device.type == 'cuda') and (ngpu > 1):

    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights

#  to ``mean=0``, ``stdev=0.02``.

netG.apply(weights_init)

# Print the model

#input=torch.randn(32,nz).to(device)

#output=netG(input)

#output.shape

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(ndf * 8, ndf * 16, 5, 2, 2, bias=True),
            # nn.BatchNorm2d(ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),

            # state size. ``(ndf*8) x 4 x 4``
            nn.Flatten(start_dim=1),
            nn.Linear(ndf*8*seed_size*seed_size,1,bias=True),
            # nn.Conv2d(ndf * 8, 1, 5, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# summary(netG, (128, ), batch_size, device='cuda')
# summary(netD, (1, 256, 256), batch_size, device='cuda')


# exit(0)
# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
   netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)
# Print the model
num_gen_params=sum(p.numel() for p in netG.parameters() if p.requires_grad)
num_disc_params=sum(p.numel() for p in netD.parameters() if p.requires_grad)
#print("ngp",num_gen_params)
#print("ndp",num_disc_params)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D

optimizerD = optim.Adam(netD.parameters(), lr=lr,betas=(beta1,0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Commented out IPython magic to ensure Python compatibility.
# Training Loop
# Lists to keep track of progress

img_list = []
G_losses = []
D_losses = []
iters = 0
D_x_latest=None
D_G_z1_latest=None
D_G_z2_latest=None
print("Device: ", device, "aka", torch.cuda.get_device_name(device))

def save_plot(D_losses, G_losses, i):
    x = np.arange(len(D_losses))
    plt.plot(x, D_losses , color = 'green', label="D")
    plt.plot(x, G_losses[::2] , color = 'red', label='G')
    plt.title('Loss Graph')
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"/home/cudaq/files/workspace/images/new_june/loss_update/g{i}")

import time
t_start = time.time()
# For each epoch
# print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")

for epoch in range(num_epochs):
    # For each batch in the dataloader
    print()
    print("epoch:",epoch,"\t time:", time.time()-t_start);t_start = time.time()
    for i, data in enumerate(train_dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device, non_blocking = True)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        ## Train with all-fake batch
        # Generate batch of latent vectors
        #noise = torch.randn(b_size, nz,device=device)
        noise=np.random.uniform(-1, 1, size=(b_size, nz))
        noise = torch.from_numpy(noise).float().to(device, non_blocking = True)
        #noise=torch.empty(b_size, nz, 1, 1, device=device).normal_(0, 1)
        # Generate fake image batch with G
        fake = netG(noise)
        #label.fill_(fake_label)
        label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        D_x_latest=D_x
        D_G_z1_latest=D_G_z1
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        D_losses.append(errD.item())
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        for j in range(2):
          netG.zero_grad()
          #label.fill_(real_label)  # fake labels are real for generator cost
          label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
          #noise = torch.randn(b_size, nz,device=device)
          noise=np.random.uniform(-1, 1, size=(b_size, nz))
          noise = torch.from_numpy(noise).float().to(device, non_blocking = True)
        # Since we just updated D, perform another forward pass of all-fake batch through D
          fake = netG(noise)
          output = netD(fake).view(-1)
        # Calculate G's loss based on this output
          errG = criterion(output, label)
        # Calculate gradients for G
          errG.backward(retain_graph=True if j==0 else False)
        #   D_G_z2 = output.mean().item()
        #   D_G_z2_latest=D_G_z2
        # Update G
          optimizerG.step()
          G_losses.append(errG.item())    
        # Output training stats
    
        if i % 50 == 0:
            # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #        % (epoch, num_epochs, i, len(train_dataloader),
            #          D_losses[len(D_losses)-1],G_losses[len(G_losses)-1], D_x_latest, D_G_z1_latest, D_G_z2_latest))
            print("epoch:", epoch)
        iters += 1
        # Check how the generator is doing by saving G's output on fixed_noise

        # Save Losses for plotting later

        ##G_losses.append(errG.item())

        #D_losses.append(errD.item()) 
        # print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e6} MB", end= '\t')
        # print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1e9} GB")
    if epoch%5==0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        vutils.save_image(grid, f"workspace/images/new_june/image_{epoch}.png")
        save_plot(D_losses, G_losses, epoch)


# training code works, but takes too much time

#save final generator and discriminator models into colab for download

torch.save(netG.state_dict(), './gen_rand_rotation.pth')

torch.save(netD.state_dict(), './dis_rand_rotation.pth')

#save final discriminator and generator loss values , and save generated images


# import pickle

# with open('D_losses_mvtec_ad_pytorch_attempt.pkl', 'wb') as f:

#     pickle.dump(D_losses, f)

# with open('G_losses_mvtec_ad_pytorch_attempt.pkl', 'wb') as f:

#     pickle.dump(G_losses, f)

#save img_list(generated imges) for download

#import pickle

# with open('img_list_mvtec_ad_pytorch_attempt.pkl', 'wb') as f:

#     pickle.dump(img_list, f)
# print("model saved")