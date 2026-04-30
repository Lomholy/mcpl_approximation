import numpy as np
import matplotlib.pyplot as plt
from vae_definition import VAE
import torch
import argparse
import sys

sys.path.append("..")
from data_load import postprocess_nn, save_data_as_mcpl
from plotting import plot_correlations_7d


# ==============================================================================
# ================   Plot the loss over iteration
# ==============================================================================


def plot_losses(
    filename_train="vae_train_losses.npy", filename_val="vae_val_losses.npy"
):
    fig, ax = plt.subplots()
    train_losses = np.load(filename_train)
    ax.plot(train_losses, label="Train loss")
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
parser.add_argument("--plot", action="store_true")
parser.add_argument("--loss", action="store_true")
args = parser.parse_args()
filename = args.correlations_file


device = "mps"

ckpt = torch.load(filename, map_location=device)

vae = VAE().to(device)
vae.load_state_dict(ckpt["state_dict"])
vae.eval()
lims = np.load("../limits.npy")
mins = torch.asarray(lims[0])
maxs = torch.asarray(lims[1])
        
with torch.no_grad():
    # Only include samples if they are within the limits of the original data

    n_samples = 1_000_00
    samples = []
    while len(samples) < n_samples:
        print(len(samples))
        # Limit samples to within the preprocessed data limits
        z = torch.randn(10_000, vae.latent_dim).to(device)
        batch = torch.asarray(vae.decode(z).cpu())
        batch = postprocess_nn(batch, filename="normalization_params.npy")
        mask = (batch >= mins) & (batch <= maxs)
        batch = batch[mask.all(axis=1)]
        samples = samples + batch.tolist()
    samples = samples[:n_samples]
    samples = torch.asarray(samples)


if args.loss:
    plot_losses()
if args.plot:
    plot_correlations_7d(samples, "Synthetic VAE: correlations postprocessed")

# save_data_as_mcpl(samples, "../vae_samples")


plt.show()
