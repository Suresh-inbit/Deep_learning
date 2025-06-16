"""
Training script for GAN using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append("..")
from models.generator_cnn import Generator_512  
from models.generator import Generator
# from models.discriminator import Discriminator
from models.discriminator_cnn import Discriminator_512
import torchvision
from torchvision4ad.datasets import MVTecAD
torch.manual_seed(99)
def train_gan_cnn(epochs=50, batch_size=1, noise_dim=128, lr=0.0002):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.CenterCrop((512,512)),
        transforms.Normalize([0.5], [0.5]), # normalize image with mean and standard deviation.
        # transforms.Resize((512, 512)),  # Reduce image size to save memory
    ])

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:

            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    dataloader = DataLoader(MVTecAD("MVTec", 'grid', train =True, transform=transform, download=True),
                          batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    sample_img = next(iter(dataloader))[0].to(device)
    img_shape = dataloader.dataset[0][0].shape
    fixed_noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
    print("size of images: ", sample_img.shape)
    # exit(0)
    # Models
    netG = Generator_512().to(device)
    netD = Discriminator_512().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # print(netG)
    # print("Parameter dtype: ",next(netG.parameters()).dtype)
    # print("number of parameters in generator: ", sum(p.numel() for p in netG.parameters()), f"| Memory occupied: {sum(p.numel() for p in netG.parameters())/(2**18)} MB")
    # print("number of parameters in D: ", sum(p.numel() for p in netD.parameters()),  f"| Memory occupied: {sum(p.numel() for p in netD.parameters())/(2**18)} MB")

    # print("Noise shape: ",fixed_noise.shape)
    # pred = netG(fixed_noise)
    # print("pred shape: ", pred.shape)

    # out = netD(sample_img)
    # print("out shape: ", out.shape, "Output : ", out[0])
    # print(netD)
    # exit(0)

    loss = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    D_losses = []
    G_losses = []
    # img_list = []
    #Training Loop
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
            # optimizerD.step()
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
                print( f"Epoch :{epoch} [{i}/264] LOSS: {error_G} , {error_D}")
            D_losses.append(error_D.item())
            G_losses.append(error_G.item())

        with torch.no_grad():
            pred = netG(fixed_noise).detach().cpu()
        img_save =torchvision.utils.make_grid(fake, padding=2,nrow=4, normalize=True)
        torchvision.utils.save_image(img_save, f"./images/mv/img_{epoch}.png")
    torch.save(netG.state_dict(), "./gen.pth")
    torch.save(netD.state_dict(), "./Des.pth")
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(np.arange(len(G_losses)), G_losses ,label="G")
    plt.plot(np.arange(len(D_losses)), D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig("./images/graph.png")
    
    # plt.imshow(img_list[0].cpu())
    # plt.show()
    # i=0
    # for img in img_list:
    #     torchvision.utils.save_image(img, f"./images/mv/img_{i}.png")


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
    noise_dim = 100
    generator = Generator(noise_dim, (1, 28, 28)).to(device)
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
    generator.eval()
    with torch.no_grad():
        z = torch.randn(25, noise_dim, device=device)
        gen_imgs = generator(z)
        save_image(gen_imgs.data, "images/generated_images.png", nrow=5)
        plot_images(gen_imgs, 5)
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
    train_gan_cnn(epochs=50)
    # test_gan()
