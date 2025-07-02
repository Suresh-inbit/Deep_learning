import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
import json # New: For saving training parameters in a structured format
from torchvision.utils import save_image, make_grid # New: For better image saving
t = datetime.datetime.now()
fname = t.strftime("%Y%m%d_%H%M%S") # e.g., "20250625_103045"
date = t.strftime("%d")
log_dir = f"logs/Jun{date}/{fname}/"
def Logger(Epochs, batch_size, lr_G, lr_D, numParmsG, numParmsD, netG, noise_dim, ngf, ndf, time_taken, losses, Comments, save_interval=50):
    """
    Logs GAN training parameters, losses, and generated images.

    Args:
        fold (str): The main folder name for logging.
        fname (str): The specific file/run name within the fold.
        Epochs (int): Total number of training epochs.
        batch_size (int): Batch size used for training.
        lr_G (float): Learning rate for the Generator.
        lr_D (float): Learning rate for the Discriminator.
        numParmsG (int): Number of parameters in the Generator.
        numParmsD (int): Number of parameters in the Discriminator.
        netG (torch.nn.Module): The Generator network.
        noise_dim (int): Dimension of the noise vector.
        ngf (int): Feature map multiplier for the Generator.
        ndf (int): Feature map multiplier for the Discriminator.
        time_taken (float): Total time taken for training (in seconds).
        losses (tuple): A tuple containing (G_losses_list, D_losses_list).
        Comments (str): Any additional comments about the training run.
        save_interval (int, optional): Interval for saving intermediate generated images. Defaults to 50.
    """
    # Use f-strings for folder paths for better readability
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Store fold globally for helper functions (consider passing as argument if preferred)
    # globals()['_current_log_fold'] = 
    # globals()['_current_log_fname'] = fname

    # t = datetime.datetime.now()

    # --- Improvement 1: Save parameters in JSON for easier parsing ---
    training_params = {
        "date": t.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": Epochs,
        "batch_size": batch_size,
        "noise_dimension": noise_dim,
        "learning_rate_G": lr_G,
        "learning_rate_D": lr_D,
        "num_parameters_G_M": round(numParmsG / 1e6, 2),
        "num_parameters_D_M": round(numParmsD / 1e6, 2),
        "feature_multiplier_G": ngf,
        "feature_multiplier_D": ndf,
        # "max_g_loss": round(max(losses[0]), 5) if losses[0] else None, # Handle empty loss list
        "time_taken_seconds": round(time_taken, 5),
        "comments": Comments
    }

    with open(f"{log_dir}training_parms.json", 'w') as f:
        json.dump(training_params, f, indent=4)

    # --- Original TXT for quick glance (optional, but good for summary) ---
    txt = f"""DATE : {t.strftime("%Y-%m-%d %H:%M:%S")} 
Number of Epochs: {Epochs} | Batch size: {batch_size}
Noise /Latent Dimension : {noise_dim}
Learning Rate:        G {lr_G}, \tD {lr_D}
Number of parameters: G {numParmsG/1e6:.2f}M, \tD {numParmsD/1e6:.2f}M
Feature Multiplier:   G {ngf}    , \tD: {ndf}

Time taken: {time_taken:.5f}s

Comments:
        {Comments}

netG: 
{netG}
    """
    with open(f"{log_dir}training_summary.txt", 'w') as f:
        f.write(txt)

    # --- Call helper functions ---
    if losses!=None:
        plot_graph(losses, numParmsG, numParmsD, iter='all')
        save_generated_image(log_dir, netG, noise_dim)
        save_numpy(log_dir, losses)

    # --- New Feature: Save intermediate generated images (if losses is long enough) ---
    # This requires `save_image_intermediate` to be called during training loop
    # The `save_interval` here is just a parameter for the logger setup.
    # The actual saving logic would be in your training loop.
    print(f"Logging complete for run: {fname}")
    print(f"Logs saved to: {os.path.abspath(log_dir)}")


def save_numpy(log_dir, losses):
    """
    Saves generator and discriminator losses as NumPy arrays.

    Args:
        log_dir (str): The directory where logs are saved.
        losses (tuple): A tuple containing (G_losses_list, D_losses_list).
    """
    g_losses, d_losses = losses
    np.save(f"{log_dir}loss_g.npy", np.array(g_losses))
    np.save(f"{log_dir}loss_d.npy", np.array(d_losses))
    print(f"Losses saved to {log_dir}loss_g.npy and {log_dir}loss_d.npy")


def plot_graph(losses, numParmsG, numParmsD, iter, Final=False):
    """
    Plots and saves the Generator and Discriminator loss graph.

    Args:
        log_dir (str): The directory where logs are saved.
        losses (tuple): A tuple containing (G_losses_list, D_losses_list).
        numParmsG (int): Number of parameters in the Generator.
        numParmsD (int): Number of parameters in the Discriminator.
    """
    if losses!=None:
        plot_dir = log_dir
        os.makedirs(plot_dir, exist_ok=True)
        if not Final:
            plot_dir = f"{log_dir}plots/"
        os.makedirs(plot_dir, exist_ok=True)
        G_losses, D_losses = losses
        plt.figure(figsize=(12, 6)) # Increased figure size
        plt.title(f"Generator and Discriminator Loss During Training\nD: {numParmsD/1e6:.2f}M, G: {numParmsG/1e6:.2f}M epoch:{iter}")
    
    # --- Improvement 2: Smoother plotting if data is noisy (optional) ---
    # You might want to apply a moving average for very noisy loss curves
    # from scipy.ndimage import uniform_filter1d
    # G_losses_smooth = uniform_filter1d(G_losses, size=10) # Adjust window size
    # D_losses_smooth = uniform_filter1d(D_losses, size=10)
    # plt.plot(np.arange(len(G_losses)), G_losses_smooth, label="G (Smoothed)")
    # plt.plot(np.arange(len(D_losses)), D_losses_smooth, label="D (Smoothed)")

    plt.plot(np.arange(len(G_losses)), G_losses, label="G Loss", alpha=0.8) # Added alpha for slight transparency
    plt.plot(np.arange(len(D_losses)), D_losses, label="D Loss", alpha=0.8)

    plt.xlabel("Iterations") # Changed to "Iterations" as losses are typically logged per batch/iteration
    plt.ylim(bottom=-1) # Ensure y-axis starts from 0 for loss
    # plt.ylim(top=10) # Keep if you want to cap the top, but can obscure high losses
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7) # Added a grid for better readability
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(f"{plot_dir}loss_graph_{iter}.png")
    plt.close() # Close the plot to free up memory
    print(f"Loss graph saved to {log_dir}loss_graph.png")


def save_generated_image(log_dir, netG, noise_dim, device="cuda"):
    """
    Generates and saves a single sample image from the trained Generator.

    Args:
        log_dir (str): The directory where logs are saved.
        netG (torch.nn.Module): The Generator network.
        noise_dim (int): Dimension of the noise vector.
        device (str, optional): The device to run the generator on. Defaults to "cuda".
                                This should match where netG is currently.
    """
    # Ensure netG is in evaluation mode
    netG.eval()
    try:
        # --- Improvement 3: Handle device placement gracefully ---
        # Get the actual device of the model if not explicitly passed
        # This assumes netG has at least one parameter
        model_device = next(netG.parameters()).device if hasattr(netG, 'parameters') and len(list(netG.parameters())) > 0 else torch.device("cpu")
        
        noise = torch.randn(1, noise_dim, 1, 1).to(model_device)
        with torch.no_grad(): # Disable gradient calculation for inference
            image = netG(noise).cpu() # Move to CPU for saving

        # --- Improvement 4: Use torchvision.utils.save_image for better handling of image formats ---
        # This is more robust than plt.imsave for common image tasks, especially if output is multichannel
        # Assumes image is in [C, H, W] or [N, C, H, W] format,
        # and values are in [0, 1] or [-1, 1] (will normalize if needed by save_image)
        # If your GAN outputs grayscale and `image.squeeze()` makes it [H, W], you might need to adjust.
        # For a single grayscale image, you might need `image.squeeze(0)` to remove batch dim, and then `image.unsqueeze(1)` to add a channel dim if save_image expects C,H,W
        
        # If the output is a single grayscale image (H, W), make it (1, H, W) for save_image
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 4 and image.shape[1] == 1: # If it's (N, 1, H, W) for grayscale
            image = image.squeeze(0) # Remove batch dim

        save_image(image, f"{log_dir}generated_sample_image.png", normalize=True, value_range=(-1, 1)) # Normalize if output is -1 to 1
        print(f"Sample generated image saved to {log_dir}generated_sample_image.png")

    except Exception as e:
        print(f"Warning: Could not save generated image. Error: {e}")
        print("Please ensure netG is correctly defined and its output format is compatible.")
    finally:
        netG.train() # Set netG back to training mode


# --- New Feature: Save intermediate generated images during training ---
# This function would be called inside your training loop, e.g., every `save_interval` iterations
def save_image_intermediate(iteration, netG, noise_dim, num_samples=9, device="cuda"):
    """
    Saves a grid of generated images at a specific iteration.
    Requires `_current_log_fold` and `_current_log_fname` to be set by Logger().

    Args:
        iteration (int): The current training iteration.
        netG (torch.nn.Module): The Generator network.
        noise_dim (int): Dimension of the noise vector.
        num_samples (int, optional): Number of images to generate in the grid. Defaults to 8.
        device (str, optional): The device to run the generator on. Defaults to "cuda".
    """
    img_dir=f"{log_dir}images/"
    os.makedirs(img_dir, exist_ok=True)
    noise = torch.randn(num_samples, noise_dim, 1, 1, device=device)
    with torch.no_grad():
        pred = netG(noise).detach().cpu()
    img_save =make_grid(pred, padding=2,nrow=int(num_samples**0.5), normalize=True)
            # img_save = fake[0]
    save_image(img_save, f"{img_dir}image{iteration}.png")

# --- New Feature: Save Model Checkpoints ---
# This function would also be called inside your training loop
def save_checkpoint(iteration, netG, netD, optimizerG, optimizerD):
    """
    Saves a checkpoint of the Generator, Discriminator, and their optimizers.

    Args:
        iteration (int): The current training iteration.
        netG (torch.nn.Module): The Generator network.
        netD (torch.nn.Module): The Discriminator network.
        optimizerG (torch.optim.Optimizer): The Generator's optimizer.
        optimizerD (torch.optim.Optimizer): The Discriminator's optimizer.
        log_dir (str): The base directory for the current log run (e.g., "logs/fold_name/fname/")
    """
    checkpoint_dir = f"{log_dir}checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = f"{checkpoint_dir}checkpoint_iter_{iteration:06d}.pth"
    torch.save({
        'iteration': iteration,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at iteration {iteration} to {checkpoint_path}")

# # Example Usage (how you would call this in your training script):
# if __name__ == "__main__":
#     # Dummy data for demonstration
#     class DummyNet(torch.nn.Module):
#         def __init__(self, output_channels=3):
#             super().__init__()
#             self.linear = torch.nn.Linear(100, 64*64*output_channels) # For 64x64 images
#             self.output_channels = output_channels
            
#         def forward(self, x):
#             # Simulate a simple generator output (e.g., 64x64 color image)
#             # Reshape to (Batch, Channels, Height, Width)
#             return self.linear(x.squeeze()).view(x.size(0), self.output_channels, 64, 64)

#     # Assume these are obtained from your training loop
#     dummy_netG = DummyNet(output_channels=3).to('cpu') # Or 'cuda' if you have one
#     dummy_netD = DummyNet(output_channels=1).to('cpu') # Discriminator output is usually 1 for real/fake

#     # Calculate number of parameters
#     num_params_G = sum(p.numel() for p in dummy_netG.parameters() if p.requires_grad)
#     num_params_D = sum(p.numel() for p in dummy_netD.parameters() if p.requires_grad)

#     g_losses_example = np.random.rand(500) * 5 + 1 # Example G losses
#     d_losses_example = np.random.rand(500) * 3 + 0.5 # Example D losses

#     # Call the main Logger function at the end of training
#     Logger(
#         fold="my_gan_project",
#         fname="run_001_dcgan_mnist",
#         Epochs=100,
#         batch_size=64,
#         lr_G=0.0002,
#         lr_D=0.0002,
#         numParmsG=num_params_G,
#         numParmsD=num_params_D,
#         netG=dummy_netG,
#         noise_dim=100,
#         ngf=64,
#         ndf=64,
#         time_taken=3600.5678,
#         losses=(g_losses_example, d_losses_example),
#         Comments="First DCGAN run on MNIST dataset with standard settings. Trying to achieve stable training.",
#         save_interval=50 # This parameter is passed, but its effect needs to be implemented in your training loop
#     )

#     # Example of how you would call save_image_intermediate and save_checkpoint
#     # These would go inside your actual training loop:
#     # Let's simulate a loop:
#     print("\nSimulating intermediate saves...")
#     current_log_dir = f"logs/my_gan_project/run_001_dcgan_mnist/"
#     dummy_optimizerG = torch.optim.Adam(dummy_netG.parameters(), lr=0.0002)
#     dummy_optimizerD = torch.optim.Adam(dummy_netD.parameters(), lr=0.0002)

#     for i in range(1, 101): # Simulating 100 iterations
#         if i % 20 == 0: # Save intermediate image every 20 iterations
#             save_image_intermediate(i, dummy_netG, noise_dim=100, num_samples=16)
#         if i % 50 == 0: # Save checkpoint every 50 iterations
#             save_checkpoint(i, dummy_netG, dummy_netD, dummy_optimizerG, dummy_optimizerD, current_log_dir)

#     print("\nDemonstration complete. Check the 'logs' folder.")