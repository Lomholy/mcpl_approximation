import cProfile
import pstats
import sys
sys.path.append("..")
from data_load import load_mcpl_file, preprocess_nn
import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
from vae_definition import VAE, vae_loss
import argparse
import time
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity, record_function

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ==============================================================================
# ===================== ARGUMENT PARSING ==================================
# ==============================================================================


def add_arguments():
    # First parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--n_particles",
            default=1e+5,
            help="Number of particles used in training the variational auto encoder")

    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )

    parser.add_argument(
        "--kl_weight", type=float, default=0.2, help="KL divergence weight"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (e.g., cpu, mps or cuda)",
    )
    return parser


# ==============================================================================
# ===================== Single training run ==================================
# ==============================================================================


def train_vae(
    train_loader,
    hyperparameters,
    device,
    filename="vae_best.pth"
):
    start = time.time()
    best_val = float("inf")
    train_losses = []

    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=hyperparameters["lr"])
    for epoch in range(hyperparameters["epochs"]):

        kl_weight = hyperparameters["kl_weight"]
        vae.train()
        for (x,) in train_loader:
            x = x.to(device)
            recon, mu, sig = vae(x)
            loss, recon_loss, kl_loss = vae_loss(
                recon, x, mu, sig,
                kl_weight=kl_weight, current_epoch=epoch,
                total_epochs=hyperparameters["epochs"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        vae.eval()
        epoch_time = time.time()
        print(f"Epoch {epoch}\ttrain={loss.item():.4g}\tBest loss = {best_val:.4g}")
        print(f"Latent loss = {kl_loss:.4g}\t reconstruction loss {recon_loss:.4g}")
        print(f"Time elapsed {epoch_time - start:.1f}\n")

        if (loss < best_val or epoch < 2/3 * hyperparameters["epochs"]):
            best_val = loss
            torch.save(
                {
                    "state_dict": vae.state_dict(),
                    "hyperparameters": hyperparameters,
                },
                filename,
            )
    return train_losses

# ==============================================================================
# =================== END OF FUNCTION DEFINITIONS ==============================
# ==============================================================================



if __name__ == "__main__":
    parser = add_arguments()
    args = parser.parse_args()
    device = args.device
    batch_size = args.batch_size
    n_particles = int(args.n_particles)
    hyperparameters = {
        "epochs": args.epochs,
        "kl_weight": args.kl_weight,
        "lr": args.lr,
    }
    data = load_mcpl_file("../ODIN.mcpl.gz", n_particles)
    data, mins, dxs = preprocess_nn(data)
    # Save normalization params
    np.save("normalization_params.npy", np.array([mins, dxs]))
    with cProfile.Profile() as prof:
        dataset = TensorDataset(data)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_losses = train_vae(
                train_loader, hyperparameters, device=device
            )
        np.save("vae_train_losses.npy", train_losses)
        stats = pstats.Stats(prof)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats("profile_results.prof")
        # stats.print_stats()
