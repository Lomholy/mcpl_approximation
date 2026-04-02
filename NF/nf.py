import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("..")
from data_load import load_mcpl_file, preprocess_nn
from nf_definition import VelocityField
import argparse
import torch
import numpy as np


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================


def add_arguments(parser):
    parser.add_argument("--n_particles", default=1e5)
    parser.add_argument("--model_filename", default="flowModel.pth")
    parser.add_argument("--device", default="mps")


# ---------------------------------------------------------
#  Weighted Flow Matching loss
# ---------------------------------------------------------


def weighted_flow_matching_loss(model, x0, x1, t, w):
    """
    x1: (B,5) data samples
    w: (B,) importance weights, >=0
    """

    # Interpolate path
    xt = (1 - t) * x0 + t * x1

    # True velocity for straight-line FM
    v_target = x1 - x0

    # Model velocity
    v_pred = model(xt, t)
    # Weighted MSE
    loss = ( w * ((v_pred - v_target) ** 2).mean(dim=-1)).mean()

    return loss


# ---------------------------------------------------------
#  Training loop
# ---------------------------------------------------------
def train(
        model,
        dataset,
        weights,
        filename,
        device="mps",
        steps=100000,
        lr=1e-3,
        batch_size=10000
        ):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_state = None

    losses = []

    N = dataset.shape[0]
    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,))
        x0 = torch.randn(batch_size, dataset.shape[1]).to(device)
        x1 = dataset[idx].to(device)
        t = torch.rand(batch_size, 1).to(device)
        w = weights[idx].to(device)
        loss = weighted_flow_matching_loss(model, x0, x1, t, w)

        # Make the loss, an average of more than 1 step
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            print(f"Saving the best model at step {step}, with loss {best_loss:.2g}")
            best_state = {"state_dict": model.state_dict(),
                          "loss": loss.item()}
            torch.save(best_state, filename)
        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.4g}")

    return losses


# ---------------------------------------------------------
# USAGE
# ---------------------------------------------------------


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
data = load_mcpl_file("../ODIN.mcpl.gz", int(args.n_particles))
data, weights, mins, dxs = preprocess_nn(data)
total_int = weights.sum()
weights = weights / total_int
dim = data.shape[1]
device = args.device

print("Weights:", weights.min(), weights.max(), weights.mean(), weights.std())

model = VelocityField(network_dimensions=512, layers=5).to(device)

losses = train(model, data, weights, args.model_filename)

np.save("losses.npy", np.array(losses))

