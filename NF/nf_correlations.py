import sys
sys.path.append("..")
from plotting import plot_correlations_7d
from data_load import postprocess_nn, save_data_as_mcpl
from nf_definition import VelocityField
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np


def sample_flow(model, n_samples, device="mps", t_steps=1000):
    """
    Evolve x0 through the learned flow to produce a sample from p1.
    """
    xt = torch.randn(n_samples, 6, device=device)
    with torch.no_grad():
        for i, t_val in enumerate(torch.linspace(0, 1, t_steps, device=device), start=1):
            if i % 100 == 0:
                print(f"Generating samples. At time step {i}")
            pred = model(xt, t_val.expand(n_samples, 1))
            xt = xt + (1 / t_steps) * pred
    return xt


parser = argparse.ArgumentParser()

parser.add_argument("--plot", action="store_true")
parser.add_argument("--loss", action="store_true")
parser.add_argument("--n_samples", default=1_000_000)
args = parser.parse_args()
batch_size = 1000
n_samples = int(args.n_samples)
device = "mps"


ckpt = torch.load("flowModel.pth", map_location=device)
loss = ckpt["loss"]
print(f"Loaded model showed a loss of {loss} on its batch")

nf = VelocityField().to(device)
nf.load_state_dict(ckpt["state_dict"])
nf.eval()

lims = np.load("../limits.npy") 


mins = torch.asarray(lims[0], dtype=torch.float32, device="cpu")
maxs = torch.asarray(lims[1], dtype=torch.float32, device="cpu")


samples = []
while len(samples) < n_samples:
    print(len(samples))
    # Limit samples to within the preprocessed data limits
    batch = sample_flow(nf, min([100_000, n_samples])).cpu()
    batch = postprocess_nn(batch, filename="normalization_params.npy")
    mask = (batch >= mins) & (batch <= maxs)
    batch = batch[mask.all(axis=1)]
    samples = samples + batch.tolist()
samples = samples[:n_samples]
samples = torch.asarray(samples)

print(f"Actually plotted samples = {samples.shape[0]}")
if args.plot:
    plot_correlations_7d(samples, "Synthetic CFM: correlations postprocessed")

save_data_as_mcpl(samples, "../cmf_samples")
plt.show()
