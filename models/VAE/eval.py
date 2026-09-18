import numpy as np
import matplotlib.pyplot as plt
from model import VAE, Sampler
import torch
import argparse
import sys

sys.path.append("../../utils/")
from plotting import plot_correlations_7d
from data_load import export_model_as_onnx, export_model_as_torchscript


# ==============================================================================
# ================   Plot the loss over iteration
# ==============================================================================


def plot_losses(
    filename_train="vae_train_losses.npy", filename_val="vae_val_losses.npy"
):
    fig, ax = plt.subplots()
    train_losses = np.load(filename_train)
    val_losses = np.load(filename_val)
    print(val_losses)
    ax.plot(val_losses[:,0], train_losses, ".", label="Train loss")
    ax.plot(val_losses[:,0], val_losses[:,1], ".", label="Validation loss")

    ax.set(
        title="Loss per iteration",
        xlabel="Iteration",
        ylabel="Loss [reconstruction + Kullback Liebler div]",
    )
    ax.legend()
    return fig, ax


# ==============================================================================
# ================ Make N synthetic samples, and plot their correlations
# ==============================================================================


parser = argparse.ArgumentParser()

parser.add_argument("--correlations_file", type=str, default="../../data_files/models/vae.pth")
parser.add_argument("--device", type=str, default="mps")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--loss", action="store_true")
parser.add_argument("--n_samples", default=1_000_000)
args = parser.parse_args()
filename = args.correlations_file
plot = args.plot
device = args.device
n_samples = int(args.n_samples)


ckpt = torch.load(filename, map_location=device)

vae = VAE().to(device)
vae.load_state_dict(ckpt["state_dict"])
vae.eval()
sampler = Sampler(vae, device=device)
with torch.no_grad():
    # Only include samples if they are within the limits of the original data

    samples = []
    while len(samples) < n_samples:
        print(len(samples))
        # Sampler expects a 12-wide Gaussian buffer (matching Source_ML(_torch).comp);
        # it only uses the first lat_dim columns internally.
        x = torch.randn(10_000, 12, device=device)
        batch = torch.asarray(sampler.forward(x).cpu())
        samples = samples + batch.tolist()
    samples = samples[:n_samples]
    samples = torch.asarray(samples)

export_model_as_onnx(sampler, "../../data_files/models/VAE_sampler.onnx", device)
export_model_as_torchscript(sampler, "../../data_files/models/VAE_sampler.pt", device)

if plot:
    plot_losses(filename_train="../../data_files/losses/vae_train.npy", filename_val="../../data_files/losses/vae_val.npy")
    plot_correlations_7d(samples, "Synthetic VAE: correlations postprocessed")

torch.save(samples, "../../data_files/samples/VAE_gauss.pkl")


plt.show()
