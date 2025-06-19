import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
def Logger(fold,fname,Epochs,batch_size, lr_G, lr_D, numParmsG, numParmsD, netG, noise_dim, ngf, ndf , time, losses, Comments):
    globals()['fold']=fold
    t = datetime.datetime.now()
    txt = f"""DATE : {t} \n
Number of Epochs: {Epochs} | Batch size: {batch_size}
Noise /Latent Dimension : {noise_dim}
Learning Rate:        G {lr_G}, \tD {lr_D}
Number of parameters: G {numParmsG/1e6}M, \tD {numParmsD/1e6}M
Feature Multiplier:   G {ngf}    , \tD: {ndf}
Maximum G Loss: {max(losses[0]): .5f}
Time taken: {time:.5f}s

Comments:
        {Comments}

    """
    os.makedirs(f"logs/{fold}/", exist_ok=True)
    os.makedirs(f'logs/{fold}/{fname}/', exist_ok=True)
    with open(f"logs/{fold}/{fname}/training_parms.txt", 'w') as f:
        f.write(txt)
    plot_graph(fold, f"{fname}/graph", *losses, numParmsG, numParmsD)
    save_generated_image(fname, netG, noise_dim)
    save_numpy(fname, losses)

def save_numpy(fname, losses):
    g , d = losses
    with open(f"logs/{fold}/{fname}/loss_d.npy", 'wb') as f:
        np.save(f, np.array(d))
    with open(f"logs/{fold}/{fname}/loss_g.npy", 'wb') as f:
        np.save(f, np.array(g))

def plot_graph(fold,fname,G_losses, D_losses, numParmsG, numParmsD):
    """
    Args:
        Folder: str
        File_name : str
        Gen_Loss : np.array()
        Dis_Loss : np.array()
        numparmsG: int
        numparmsD: int
    """
    plt.figure(figsize=(10,5))
    plt.title(f"Generator and Discriminator Loss During Training. D: {numParmsD//1e6}M,  G: {numParmsG//1e6}M")
    plt.plot(np.arange(len(G_losses)), G_losses ,label="G")
    plt.plot(np.arange(len(D_losses)), D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(f"./logs/{fold}/{fname}.png")

def save_generated_image(fname, netG, noise_dim):
    noise = torch.randn(1, noise_dim, 1, 1).to('cuda:1')
    image = netG(noise)
    plt.imsave(f"logs/{fold}/{fname}/Generated_image.png", image.cpu().squeeze().detach().numpy(), cmap = "gray")

# Logging(1,2,3,4,5,6,7)