"""
Training script for GAN using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append("..")
from utils.Dataset import Augmented
from utils.log import Logger, plot_graph
from models.generator_cnn import Generator_512 as Generator_cnn 
from models.generator import Generator
# from models.discriminator import Discriminator
from models.discriminator_cnn import Discriminator_512 as Discriminator_cnn
import torchvision
from torchvision4ad.datasets import MVTecAD
import time
date = datetime.datetime.now().day
torch.manual_seed(99)
def train_gan_cnn(epochs=50, batch_size=2, noise_dim=128, lr_G=0.0002, lr_D=0.0001, verbose = True, stop = False):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(0.6), #probability
        transforms.RandomVerticalFlip(0.6) ,
        # v2.GaussianNoise(sigma=0.04),
        transforms.Normalize([0.5], [0.5]), # normalize image with mean and standard deviation.
        transforms.Resize((512, 512)),  # Reduce image size to save memory
    ])

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:

            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    Comments = "Changed kernel size to 3 only on discriminator, increaded num features in generator"
    # dset = MVTecAD("MvTec", 'grid', train =True, transform=transform, download=True)
    # dset = ImageFolder("Datasets/MvTec/", transform=transform)
    dset = ImageFolder("Datasets/MvTec/")
    dset = Augmented(dset, transform = transform)
    dataloader = DataLoader(dset,
                          batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
    sample_img = next(iter(dataloader))[0].to(device)
    img_info =False
    if img_info:
        img_shape = dataloader.dataset[2]
        print(len(dataloader.dataset), len(dataloader))
        print(sample_img[0][0])
        # plt.imshow(img_shape[0][0])
        plt.imsave("images/image3.png", sample_img[0][0].cpu().numpy(),cmap ='gray')
        print(sample_img[0][0].cpu())
        torchvision.utils.save_image(sample_img[0][0].cpu(), "images/image.png")
    # exit(0)
    # Models
    ngf = 45
    ndf = 32
    netG = Generator_cnn(latent_dim=noise_dim, nf=ngf).to(device)
    netD = Discriminator_cnn(nf=ndf).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)
    # netD.load_state_dict(torch.load('saved_models/dis.pth', map_location = device))
    # netG.load_state_dict(torch.load('saved_models/gen.pth', map_location = device))
    fixed_noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
    numParmsG = sum(p.numel() for p in netG.parameters())
    numParmsD = sum(p.numel() for p in netD.parameters())

    # plt.imsave("images/jun19/generated_image.png", netG(fixed_noise)[0].cpu().detach().squeeze().numpy())
    if verbose:
        print(netG)
        print(netD)
        print("USING DEVICE:", torch.cuda.get_device_name(device))
        print("size of images: ", sample_img.shape)
        print("Parameter dtype: ",next(netG.parameters()).dtype)
        print("number of parameters in generator: ", numParmsG/1e6, f"| Memory occupied: {numParmsG/(2**18)} MB")
        print("number of parameters in D: ", numParmsD/1e6 ,  f"| Memory occupied: {numParmsD/(2**18)} MB")
        print("Noise shape: ",fixed_noise.shape)
        pred = netG(fixed_noise)
        out = netD(sample_img)

        print("pred shape: ", pred.shape)
        print("out shape: ", out.shape, "Output : ", out[0])
    if stop: exit(0)

    loss = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 0.9
    fake_label = 0.1

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))

    D_losses = []
    G_losses = []
    #Training Loop
    start = time.time()
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):

            # (1) Update D network: 

            netD.zero_grad()  # refresh the gradient 
            img = data[0].to(device)
            # print("Image shape: ",img.shape)
            b_size = img.size(0) # current batch size
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device) # all ones because training netD with real images
            output = netD(img).view(-1)
            # print("Output_D shape :",output.shape, 'label shape:' , label.shape)
            # exit(0)
            # print(output)
            error_real = loss(output, label)
            error_real.backward()
            noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            # print(noise.shape)
            fake = netG(noise)

            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            error_fake = loss(output, label)
            error_fake.backward()
            error_D = error_fake + error_real # for tracking D error
            optimizerD.step()

            # (2) Update G network: 
            netG.zero_grad()
            label.fill_(real_label)  #  labels are real for generator cost est
            output = netD(fake).view(-1)
            error_G = loss(output, label)
            error_G.backward()
            optimizerG.step()

            if i % 50 ==0:
                print( f"Epoch :{epoch} [{i}/256] LOSS: G{error_G:.5f} , D{error_D:.5f}")
            D_losses.append(error_D.item())
            G_losses.append(error_G.item())
        if epoch%10==0:
            with torch.no_grad():
                pred = netG(fixed_noise).detach().cpu()
            img_save =torchvision.utils.make_grid(fake[:1], padding=2,nrow=1, normalize=True)
            # img_save = fake[0]
            torchvision.utils.save_image(img_save, f"./images/jun{date}/img_K3_{epoch+100}.png")
            plot_graph('tmp', f'graph_{epoch}', G_losses, D_losses, numParmsG, numParmsD)
    torch.save(netG.state_dict(), "./saved_models/gen_k3.pth")
    torch.save(netD.state_dict(), "./saved_models/dis_k3.pth")
    total_time = time.time() - start
    print(" Total time: %.2fs" % total_time)
    Logger(f'Jun{date}', f"5_G{numParmsG//1e6}_D{numParmsD//1e6}", epochs,batch_size, lr_G, lr_D, numParmsG, numParmsD, netG, noise_dim,ngf, ndf, total_time, [G_losses, D_losses], Comments)
    # #plot loss graph
    # plt.figure(figsize=(10,5))
    # plt.title(f"Generator and Discriminator Loss During Training. D: {numParmsD//1e6}M,  G: {numParmsG//1e6}M")
    # plt.plot(np.arange(len(G_losses)), G_losses ,label="G")
    # plt.plot(np.arange(len(D_losses)), D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # # plt.show()
    # plt.savefig(f"./images/graph_{numParmsD//1e6}.png")
    
   
def train_gan(
    epochs=10,
    batch_size=64,
    noise_dim=128,
    lr=0.0002,
    device=None
):
    """
    Train a simple GAN on MNIST dataset.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("images", exist_ok=True)

    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize([0.5], [0.5]),
        # transforms.Resize((256, 256)),  # Reduce image size to save memory
    ])
    dataloader = DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )
    # print("size of images: ", dataloader.dataset[0][0].shape)
    img_shape = dataloader.dataset[0][0].shape
    print(img_shape)
    # exit(0)

    # batch_size = 2  # Reduce batch size to save memory
    # dataset = MVTecAD("MVTec", 'grid', train=True, transform=transform, download=True)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print("size of images: ", dataset[1][0].shape)
    # img_shape = dataset[0][0].shape
    # plt.imshow(dataset[0][0].numpy().squeeze(), cmap='gray')
    # plt.axis('off')
    # plt.show()
    # print("label of images: ", [i[1] for i in dataset])
    # print("size of images: ", img_shape)
    # exit(0)
    # Models

    generator = Generator(noise_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    print("number of parameters in generator: ", sum(p.numel() for p in generator.parameters()))
    # Loss and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            imgs = batch[0]
            real_imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)
            noise = torch.randn(64, 64, 1, 1, device=device)
            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), noise_dim, device=device)
            gen_imgs = generator(noise)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if i == len(dataloader) - 1:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                save_image(gen_imgs.data[:25], f"images/{epoch}_{i}.png", nrow=5)

    # Plot losses after training
    torch.save(generator.state_dict(), "generator.pth")
    plt.figure(figsize=(10,5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.savefig('images/loss_curve.png')
    plt.close()

def test_gan():
    """
    Test the trained GAN by generating images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_dim = 128
    generator = Generator_cnn(nf=170).to(device)
    generator.load_state_dict(torch.load("saved_models/gen.pth", map_location=device))
    generator.eval()
    with torch.no_grad():
        z = torch.randn(4, noise_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        save_image(gen_imgs.data, "images/generated_images.png", nrow=2)
        # plot_images(gen_imgs, 5)
        print("Generated images saved to 'images/generated_images.png'")

def save_image(tensor, filename, nrow=5):
    """
    Save a grid of images.
    """
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True)
    # ndarr = grid.mul(127.5).add(127.5).clamp(0,255).byte().cpu().numpy()
    plt.imsave(filename, np.transpose(grid.cpu().numpy(), (1,2,0)))

def plot_images(images, n_rows):
    """
    Plot a grid of images.
    """
    plt.figure(figsize=(10, 10))
    for i in range(n_rows * n_rows):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_gan_cnn(epochs=50, verbose= True, stop = True)
    # test_gan()
