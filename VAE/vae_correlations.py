import numpy as np
import matplotlib.pyplot as plt
from vae_definition import VAE
import torch
import argparse
import sys

sys.path.append("..")
from data_load import postprocess_nn
from plotting import plot_correlations_6d


# ==============================================================================
# ================   Plot the loss over iteration
# ==============================================================================


def plot_losses(
    filename_train="vae_train_losses.npy", filename_val="vae_val_losses.npy"
):
    fig, ax = plt.subplots()
    train_losses = np.load(filename_train)
    val_losses = np.load(filename_val)
    val_x = np.linspace(0, train_losses.shape[0], val_losses.shape[0])
    ax.plot(train_losses, label="Train loss")
    ax.plot(val_x, val_losses, label="Validation losses")
    ax.legend()
    ax.set(yscale="log")
    ax.set(
        title="Loss per iteration",
        xlabel="Iteration",
        ylabel="Loss [reconstruction + Kullback Liebler div]",
    )
    return fig, ax


# ==============================================================================
# ================ Make N synthetic samples, and plot their correlations
# ==============================================================================


parser = argparse.ArgumentParser()

parser.add_argument("--correlations_file", type=str, default="vae_best.pth")
args = parser.parse_args()
device = "mps"

ckpt = torch.load("vae_best.pth", map_location=device)

vae = VAE().to(device)
vae.load_state_dict(ckpt["state_dict"])
vae.eval()

with torch.no_grad():
    z = torch.randn(100000, vae.latent_dim).to(device)  # sample 1000 latent vectors
    samples = vae.decode(z)  # synthetic data in [0,1]
samples = samples.cpu()

# samples = postprocess_nn(samples, 0, 0, filename="normalization_params.npy")

plot_correlations_6d(samples, "Synthetic VAE: correaltions post")

plt.show()
