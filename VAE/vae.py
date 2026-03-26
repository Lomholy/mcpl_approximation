
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


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def add_arguments():
    # First parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_type",
        type=str,
        help="Training type. Could be either single_training or hyperparameter_scan",
        default="single_training",
    )
    parser.add_argument(
        "--n_hype_scans",
        type=int,
        default=5,
        help="Number of points in hyperparameter space that is ",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )

    parser.add_argument(
        "--kl_weight", type=float, default=2, help="KL divergence weight"
    )
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (e.g., cpu, mps or cuda)",
    )
    parser.add_argument(
        "--annealing",
        type=float,
        default=1,
        help="How many epochs the kl term is increased over",
    )
    return parser


def train_vae(
    train_loader,
    val_loader,
    hyperparameters,
    filename="vae_best.pth",
):
    start = time.time()
    best_val = float("inf")
    no_improve = 0
    train_losses = []
    val_losses = []

    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=hyperparameters["lr"])
    for epoch in range(hyperparameters["epochs"]):
        if epoch < hyperparameters["annealing"]:
            kl_weight = hyperparameters["kl_weight"] * (
                epoch / hyperparameters["annealing"]
            )
        else:
            kl_weight = hyperparameters["kl_weight"]
        vae.train()
        for (x,) in train_loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss, recon_loss, kl_loss = vae_loss(
                recon, x, mu, logvar, kl_weight=kl_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        vae.eval()
        val_losses_batch = []
        with torch.no_grad():
            for (xv,) in val_loader:
                xv = xv.to(device)
                recon, mu, logvar = vae(xv)
                vloss, _, _ = vae_loss(recon, xv, mu, logvar, kl_weight=kl_weight)
                val_losses_batch.append(vloss.item())
        val_mean = np.mean(val_losses_batch)
        val_losses.append(val_mean)
        epoch_time = time.time()
        print(f"Epoch {epoch}  train={loss.item():.4g}  val={val_mean:.4g}\tBest loss = {best_val:.4g}")
        print(f"Time elapsed {epoch_time - start}")

        if (
            val_mean < best_val
            or epoch < hyperparameters["annealing"] + hyperparameters["patience"]
        ):
            best_val = val_mean
            no_improve = 0
            torch.save(
                {
                    "state_dict": vae.state_dict(),
                    "hyperparameters": hyperparameters,
                },
                filename,
            )
        else:
            no_improve += 1
            if no_improve >= hyperparameters["patience"]:
                print("Early stopping.")
                break
    return train_losses, val_losses


def hyperparameter_sweep(
    n_scans, hyperparameter_space, train_loader, val_loader
):
    # Hyperparameter plan:
    # Build a list of hyperparameter dicts that I test
    # Then loop over each dict and train a model on it.
    # save each model, and their losses.
    hyperparameter_scan = []
    hp_space = hyperparameter_space
    for i in range(n_scans):
        scan = {}
        # Choose a random uniform value in the bounds of the hyperparameter space
        scan["kl_weight"] = np.random.uniform(
            hp_space["kl_weight"][0], hp_space["kl_weight"][1]
        )
        scan["annealing"] = int(
            np.random.uniform(hp_space["annealing"][0], hp_space["annealing"][1])
        )
        scan["lr"] = np.random.uniform(hp_space["lr"][0], hp_space["lr"][1])
        scan["epochs"] = int(
            np.random.uniform(hp_space["epochs"][0], hp_space["epochs"][1])
        )
        scan["patience"] = int(
            np.random.uniform(hp_space["patience"][0], hp_space["patience"][1])
        )
        hyperparameter_scan.append(scan)
    # Now make a training on each hyperparameter sweep
    min_loss_idx = 0
    min_loss = 1

    for i, hyperparameter in enumerate(hyperparameter_scan):
        train_losses, val_losses = train_vae(
            train_loader,
            val_loader,
            hyperparameter,
            filename=f"hype_scan_{i}.pth",
        )
        if min_loss > train_losses[-1]:
            min_loss_idx = i
            min_loss = train_losses[-1]
            train_losses_out = train_losses
            val_losses_out = val_losses

    return min_loss_idx, train_losses_out, val_losses_out


parser = add_arguments()

args = parser.parse_args()

device = args.device
batch_size = args.batch_size

hyperparameters = {
    "epochs": args.epochs,
    "kl_weight": args.kl_weight,
    "lr": args.lr,
    "patience": args.patience,
    "annealing": args.annealing,
}

data = load_mcpl_file("../ODIN.mcpl.gz", 10)

data, mins, dxs = preprocess_nn(data)
# Save normalization params
np.save("normalization_params.npy", np.array([mins, dxs]))


dataset = TensorDataset(data)
n_total = len(dataset)
n_val = int(0.1 * n_total)
n_train = n_total - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


if args.train_type == "single_training":
    train_losses, val_losses = train_vae(
        train_loader, val_loader, hyperparameters
    )

if args.train_type == "hyperparameter_scan":
    # Define the hyperparameter mins and maxes
    hyperparameter_scan = {
        "kl_weight": [0, 1],
        "epochs": [30, 60],
        "lr": [4e-3, 4e-4],
        "patience": [5, 6],
        "annealing": [5, 30],
    }
    best_idx, train_losses, val_losses = hyperparameter_sweep(
        args.n_hype_scans,
        hyperparameter_scan,
        train_loader,
        val_loader,
    )
    print(
        f"Best index was {best_idx}! final training loss was {
            train_losses[-1]:.4g}, and final validation loss was {val_losses[-1]:.4g}"
    )
np.save("vae_train_losses.npy", train_losses)
np.save("vae_val_losses.npy", val_losses)
