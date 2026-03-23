import numpy as np
import matplotlib.pyplot as plt
from vae_definition import VAE
import torch
import argparse

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


def plot_correlations(filename="vae_best.pth"):
    device = "mps"
    ckpt = torch.load(filename, map_location=device)

    vae = VAE().to(device)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()

    mins = ckpt["mins"].to(device)
    scales = ckpt["scales"].to(device)

    labels = [
        "Weight",
        "Energy",
        "Direction x",
        "Direction y",
        "Direction z",
        "Position x",
        "Position y",
    ]
    # labels = ["Angle theta", "Angle phi", "Position x", "Position y"]
    with torch.no_grad():
        z = torch.randn(100000, vae.latent_dim).to(device)  # sample 1000 latent vectors
        samples = vae.decode(z)  # synthetic data in [0,1]
        samples = samples.clamp(0, 1)
        samples = samples * scales + mins
    samples = samples.cpu()
    input_dim = 7
    fig, ax = plt.subplots(ncols=input_dim, nrows=input_dim, figsize=(15, 20))
    for i in range(input_dim):
        for j in range(input_dim):
            if j > i:
                ax[i, j].set_axis_off()
                continue
            if j == i:
                try:
                    ax[i, j].hist(samples[:, i], bins=50)
                    ax[i, j].set(xlabel=labels[i], ylabel="Counts")
                except Exception as e:
                    print(e)
                    ax[i, j].set_axis_off()
                continue
            ax[i, j].hist2d(samples[:, i], samples[:, j], bins=100)
            ax[i, j].set(xlabel=labels[i], ylabel=labels[j])
    fig.tight_layout()

parser = argparse.ArgumentParser()

parser.add_argument("--correlations_file", type=str, default="vae_best.pth")
args = parser.parse_args()
plot_losses()

plot_correlations(filename=args.correlations_file)

plt.show()
