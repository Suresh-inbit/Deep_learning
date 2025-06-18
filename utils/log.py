import os
import numpy as np
import torch
import matplotlib.pyplot as plt
def Logger(fold,fname,Epochs, lr_G, lr_D, numParmsG, numParmsD, netG, noise_dim, ngf, ndf , time, losses):
    globals()['fold']=fold
    txt = f"""Number of Epochs: {Epochs}
Noise Dimension: {noise_dim}
Learning Rate:        G {lr_G}, \tD {lr_D}
Number of parameters: G {numParmsG/1e6}M, \tD {numParmsD/1e6}M
Feature Multiplier:   G {ngf}    , \tD: {ndf}
Maximum G Loss: {max(losses[0]): .5f}
Time taken: {time:.5f}s

    """
    os.makedirs(f"logs/{fold}/", exist_ok=True)
    with open(f"logs/{fold}/training_parms_{fname}.txt", 'w') as f:
        f.write(txt)
    plot_graph(fname, *losses, numParmsD, numParmsG)
    save_generated_image(fname, netG, noise_dim)
    
def plot_graph(fname,G_losses, D_losses, numParmsD, numParmsG):
    plt.figure(figsize=(10,5))
    plt.title(f"Generator and Discriminator Loss During Training. D: {numParmsD//1e6}M,  G: {numParmsG//1e6}M")
    plt.plot(np.arange(len(G_losses)), G_losses ,label="G")
    plt.plot(np.arange(len(D_losses)), D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(f"./logs/{fold}/graph_{fname}.png")

def save_generated_image(fname, netG, noise_dim):
    noise = torch.randn(1, noise_dim, 1, 1).to('cuda:1')
    image = netG(noise)
    plt.imsave(f"logs/{fold}/Generated_image_{fname}.png", image.cpu().squeeze().detach().numpy(), cmap = "gray")

# Logging(1,2,3,4,5,6,7)