import copy
from tqdm import tqdm
import time
import argparse
from model import VAE
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import numpy as np
import sys

sys.path.append("../../utils/")
from data_load import load_mcpl_file, transform, export_model_as_onnx, export_model_as_torchscript

# ==============================================================================
# ===================== ARGUMENT PARSING ==================================
# ==============================================================================


def add_arguments():
    # First parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_particles",
        default=1e6,
        help="Number of particles used in training the variational auto encoder",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (e.g., cpu, mps or cuda)",
    )
    return parser


def sigmoid(x, top, center, slope=0.05):
    return (top) / (1 + np.exp((center - x) * slope))


def vae_loss(model, x, kl_weight, epoch, total_epochs):

    recon, mu, sig = model(x)
    if (
        torch.any(torch.isnan(recon))
        or torch.any(torch.isnan(mu))
        or torch.any(torch.isnan(sig))
    ):
        print("NaN detected! model output")
        print(recon, mu, sig)
        exit(0)
    # w = min(1.0, (epoch + 1) / (0.3 * total_epochs))
    kl = 0.1 * kl_weight  # * w
    gs = 0.1
    latent_loss = -kl * 0.5 * torch.mean(1 + sig - sig.exp() - mu**2)

    # Reconstruction loss (L1)
    rec_sig, rec_mu = torch.std_mean(recon, dim=None)
    reconstruction_loss = (
        F.mse_loss(recon, x, reduction="mean")
        # + gs*abs((rec_sig**2).log() + (recon - rec_mu)**2 / rec_sig**2).sum(dim=1).mean()
        + torch.abs(1 - rec_sig)
        #+ torch.abs(rec_mu)
    )
    # Total VAE loss
    vae_loss = reconstruction_loss + latent_loss
    if torch.isnan(reconstruction_loss) or torch.isnan(latent_loss):
        print("NaN detected!")
        print(reconstruction_loss, latent_loss)
        exit(0)

    return vae_loss, reconstruction_loss, latent_loss


def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), model.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1.0 - decay)


# ==============================================================================
# ===================== Single training run ==================================
# ==============================================================================


def train_vae(
    train_loader, val_loader, epochs, kl_weight, device, filename="../../data_files/models/vae.pth"
):
    start = time.time()
    train_losses = []
    val_losses = []

    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4)
    ema = copy.deepcopy(vae).eval().requires_grad_(False)

    for epoch in tqdm(range(epochs)):
        vae.train()
        for (x,) in train_loader:
            x = x.to(device)
            loss, recon_loss, kl_loss = vae_loss(
                vae, x, kl_weight, epoch=epoch, total_epochs=epochs
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            update_ema(ema, vae)

        epoch_time = time.time()

        if epoch % 5 == 0:
            vae.eval()
            ema.eval()
            train_losses.append(loss.item())

            with torch.no_grad():
                val_loss = 0.0
                n = 0
                for (x_val,) in val_loader:
                    x_val = x_val.to(device)
                    val_loss += vae_loss(
                        ema, x_val, kl_weight, epoch=epoch, total_epochs=epochs
                    )[0].item()
                    n += 1
                val_loss /= n

            val_losses.append([epoch, val_loss])

            print(f"Saving best model at step {epoch}")
            print(f"step {epoch}: train {loss.item():.5g}, val {val_loss:.5g}")
            print(f"Latent loss = {kl_loss:.4g}\t reconstruction loss {recon_loss:.4g}")
            print(f"Time elapsed {epoch_time - start:.1f}\n")

            torch.save(
                {
                    "state_dict": ema.state_dict(),
                    "step": epoch,
                    "val_loss": val_losses,
                },
                filename,
            )
    return vae, train_losses, val_losses


# ==============================================================================
# =================== END OF FUNCTION DEFINITIONS ==============================
# ==============================================================================


if __name__ == "__main__":
    parser = add_arguments()
    args = parser.parse_args()
    device = args.device
    n_particles = int(args.n_particles)

    batch_size = 1024
    kl_weight = 0.6
    epochs = 2

    data = load_mcpl_file("../../data_files/ODIN.mcpl.gz", n_particles)
    data = torch.asarray(transform(data, file_path="../../data_files/preprocess/gaussian_transformer.bin"), dtype=torch.float32)

    dataset = TensorDataset(data)

    # split dataset
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    vae, train_losses, val_losses = train_vae(
        train_loader, val_loader, epochs, kl_weight, device=device
    )
    np.save("../../data_files/losses/vae_train.npy", train_losses)
    np.save("../../data_files/losses/vae_val.npy", val_losses)
    export_model_as_onnx(vae, "../../data_files/models/VAE.onnx", device="mps")
    export_model_as_torchscript(vae, "../../data_files/models/VAE_torch.pt", device="mps")

