import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("..")
from data_load import load_mcpl_file, preprocess_nn, RankGaussianizer, save_data_as_mcpl
from plotting import plot_correlations_7d
from nf_definition import VelocityField, sample_t
import argparse
from tqdm import tqdm
import torch
import numpy as np
import pickle
import copy


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================


def add_arguments(parser):
    parser.add_argument("--n_particles", default=1e6)
    parser.add_argument("--model_filename", default="flowModel.pth")
    parser.add_argument("--device", default="mps")


# ---------------------------------------------------------
# Function to update weights with a moving average
# ---------------------------------------------------------

def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), model.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1.0 - decay)


# ---------------------------------------------------------
#  Flow Matching loss
# ---------------------------------------------------------


def flow_matching_loss(model, x0, x1, t):
    # Interpolate path
    xt = (1 - t) * x0 + t * x1

    # True velocity for straight-line FM
    v_target = x1 - x0

    # Model velocity
    v_pred = model(xt, t)
    # Weighted MSE
    loss = (((v_pred - v_target) ** 2).mean(dim=-1)).mean()

    return loss


# ---------------------------------------------------------
#  Training loop
# ---------------------------------------------------------
def train(
        model,
        dataset,
        filename,
        device="mps",
        steps=10_000,
        lr=1e-3,
        val_size=100000,
        batch_size=3000
        ):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    losses = []
    val_losses = []
    # Prepare validation step
    perm = torch.randperm(dataset.shape[0])
    val = dataset[perm[:val_size]].to(device)
    dataset = dataset[perm[val_size:]]
    val_x1 = val
    val_x0 = torch.randn_like(val_x1)
    val_t = torch.rand(val_x1.shape[0], 1, device=device)

    N = dataset.shape[0]
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    for step in tqdm(range(steps)):
        idx = torch.randint(0, N, (batch_size,))
        x0 = torch.randn(batch_size, dataset.shape[1]).to(device)
        x1 = dataset[idx].to(device)
        t = sample_t(batch_size, device)
        loss = flow_matching_loss(model, x0, x1, t)

        # Make the loss, an average of more than 1 step
        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, model)
        losses.append(loss.item())

        if step % 1000 == 0:
            # Check how the model performs against validation loss
            ema.eval()

            with torch.no_grad():
                val_loss = flow_matching_loss(ema, val_x0, val_x1, val_t).item()

            val_losses.append([step, val_loss])

            print(f"step {step}: train {loss.item():.5g}, val {val_loss:.5g}")

            if val_loss < best_loss:
                best_val_loss = val_loss

                print(f"Saving best model at step {step}, val loss {best_val_loss:.5g}")

                torch.save(
                    {
                        "state_dict": ema.state_dict(),
                        "step": step,
                        "val_loss": best_val_loss,
                    },
                    filename,
                )

    return losses, val_losses


# ---------------------------------------------------------
# USAGE
# ---------------------------------------------------------


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
data = load_mcpl_file("../ODIN.mcpl.gz", int(args.n_particles))
plot_correlations_7d(data, title="Raw input data", filename="raw_input.png")
transformer = RankGaussianizer()
data = torch.asarray(transformer.fit_transform(data), dtype=torch.float32)
torch.save(torch.asarray(data), "../gaussian_input")


with open('gaussian_transformer.pkl', 'wb') as outp:
    pickle.dump(transformer, outp, pickle.HIGHEST_PROTOCOL)
plot_correlations_7d(data, title="Rank Gaussianized input data", filename="gauss_input.png")

dim = data.shape[1]
device = args.device

model = VelocityField().to(device)

losses, val_losses = train(model, data, args.model_filename)

np.save("losses.npy", np.array(losses))
np.save("val_losses.npy", np.array(val_losses))

