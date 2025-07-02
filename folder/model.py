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
from torch.utils.data import DataLoader, Dataset
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from IPython.display import HTML
#import pandas as pd
# Set random seed for reproducibility
manualSeed = 999

#manualSeed = random.randint(1, 10000) # use if you want new results
print("Here we go again....")
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#from PIL imort I#cudaq.set_target

transform_test=transforms.Compose([transforms.Resize((512,512)), #for texture images like grids; modifying to grayscale here for runinng on IITM server
                                   transforms.Grayscale(num_output_channels=1),#for IITM server run
    transforms.RandomCrop([128, 128]),
                                transforms.ToTensor()])

transform = transforms.Compose([
    transforms.Resize((512,512)),  # Resize the image to 512x512
    transforms.Grayscale(num_output_channels=1), #for running on IITM server
    transforms.RandomCrop((128, 128)),  # Randomly crop a 128x128 patch    
    transforms.ToTensor()  # Convert the image to a tensor
])

#transform_test = transforms.Compose([transforms.Resize([128, 128]),
                                #transforms.ToTensor()])
mvtec_ad_train = MVTecAD('mvtec_ad', 'grid', train=True, transform=None, download=True) #make it true if new system 
# Custom dataset to ensure each object category has 10,000 patches
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
dataset = ImageFolder(root='./mvtec_ad/grid/train', transform=transform)

# Create the augmented dataset
augmented_dataset = AugmentedDataset(dataset)
#augmented_dataset = AugmentedDataset(mvtec_ad_train)

# Create a DataLoader
len(augmented_dataset)

batch_size=64

train_dataloader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)


ngpu=1

# uniform distribution in the range (-1,1) used for z
device = torch.device("cuda:2" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Number of workers for dataloader
workers = 2
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
num_epochs = 50

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
            #nn.ConvTranspose2d( nz, ngf * 8, 5, 1, 1, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            #nn.ReLU(True),
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
            # state size. ``(ndf*8) x 4 x 4``
            nn.Flatten(start_dim=1),
            nn.Linear(ndf*8*seed_size*seed_size,1,bias=True),
            #nn.Conv2d(ndf * 8, 1, 5, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

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
print("Training starts.")
import time
t_start = time.time()
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    print("epoch:",epoch,"\t time:", time.time()-t_start);t_start = time.time()
    for i, data in enumerate(train_dataloader):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
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
        noise = torch.from_numpy(noise).float().to(device)
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
          noise = torch.from_numpy(noise).float().to(device)

        # Since we just updated D, perform another forward pass of all-fake batch through D
          fake = netG(noise)
          output = netD(fake).view(-1)
        # Calculate G's loss based on this output
          errG = criterion(output, label)
        # Calculate gradients for G
          errG.backward(retain_graph=True if j==0 else False)
          D_G_z2 = output.mean().item()
          D_G_z2_latest=D_G_z2
        # Update G
          optimizerG.step()
          G_losses.append(errG.item())     

        # Output training stats
        #if i % 50 == 0:
         #   print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
          #        % (epoch, num_epochs, i, len(train_loader),
           #          D_loses[len(D_losses)-1],G_losses[len(G_losses)-1],errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                   % (epoch, num_epochs, i, len(train_dataloader),
                     D_losses[len(D_losses)-1],G_losses[len(G_losses)-1], D_x_latest, D_G_z1_latest, D_G_z2_latest))

            """print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e6} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1e9} GB");print(torch.cuda.get_device_name(noise.get_device()))
        # Save Losses for plotting later
        #G_losses.append(errG.item())
        #D_losses.append(errD.item()) """

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# training code works, but takes too much time

#save final generator and discriminator models into colab for download
torch.save(netG.state_dict(), './generator_mvtec_ad_pytorch_attempt.pth')
torch.save(netD.state_dict(), './discriminator_mvtec_ad_pytorch_attempt.pth')

#save final discriminator and generator loss values , and save generated images
import pickle
with open('D_losses_mvtec_ad_pytorch_attempt.pkl', 'wb') as f:
    pickle.dump(D_losses, f)
with open('G_losses_mvtec_ad_pytorch_attempt.pkl', 'wb') as f:
    pickle.dump(G_losses, f)

#save img_list(generated imges) for download
#import pickle
with open('img_list_mvtec_ad_pytorch_attempt.pkl', 'wb') as f:
    pickle.dump(img_list, f)


