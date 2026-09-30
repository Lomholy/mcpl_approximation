import numpy as np
import matplotlib.pyplot as plt
from model import VAE, Sampler
import torch
import argparse
import sys

sys.path.append("../../utils/")
from plotting import plot_correlations_12d
from data_load import inverse_transform, export_model_as_onnx, export_model_as_torchscript


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
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--loss", action="store_true")
parser.add_argument("--n_samples", default=1_000_000)
args = parser.parse_args()
filename = args.correlations_file
plot = args.plot
device = args.device
n_samples = int(args.n_samples)


ckpt = torch.load(filename, map_location=device)
n_training_samples = ckpt.get("n_training_samples", 1_000_000)
transformer_file_path = "../../data_files/preprocess/gaussian_transformer.bin"

vae = VAE().to(device)
vae.load_state_dict(ckpt["state_dict"])
vae.eval()
sampler = Sampler(vae, device=device)
with torch.no_grad():
    # Only include samples if they are within the limits of the original data

    gaussian_samples = []
    while len(gaussian_samples) < n_samples:
        print(len(gaussian_samples))
        # Sampler expects a 12-wide Gaussian buffer (matching Source_ML(_torch).comp);
        # it only uses the first lat_dim columns internally.
        x = torch.randn(10_000, 12, device=device)
        batch = torch.asarray(sampler.forward(x).cpu())
        gaussian_samples = gaussian_samples + batch.tolist()
    gaussian_samples = gaussian_samples[:n_samples]
    gaussian_samples = torch.asarray(gaussian_samples)

samples = torch.asarray(inverse_transform(gaussian_samples, file_path=transformer_file_path))

export_model_as_onnx(
    sampler,
    "../../data_files/models/VAE_sampler.onnx",
    device,
    transformer_file_path=transformer_file_path,
    n_training_samples=n_training_samples,
)
export_model_as_torchscript(
    sampler,
    "../../data_files/models/VAE_sampler.pt",
    device,
    n_training_samples=n_training_samples,
    transformer_file_path=transformer_file_path,
)

if plot:
    plot_losses(filename_train="../../data_files/losses/vae_train.npy", filename_val="../../data_files/losses/vae_val.npy")
    plot_correlations_12d(gaussian_samples, "VAE: Gaussian space", filename="../../figures/VAE_gauss.png")
    plot_correlations_12d(samples, "VAE: neutron phase space", filename="../../figures/VAE_neutron.png")

torch.save(gaussian_samples, "../../data_files/samples/VAE_gauss.pkl")
torch.save(samples, "../../data_files/samples/VAE.pkl")


plt.show()
