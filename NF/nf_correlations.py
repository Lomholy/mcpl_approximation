import sys, cProfile
sys.path.append("..")
from plotting import plot_correlations_5d
from data_load import postprocess_nn
from nf_definition import VelocityField
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from scipy.integrate import solve_ivp
import numpy as np
from torchdiffeq import odeint_adjoint as odeint # Use adjoint method for memory efficiency


def sample_flow(model, n_samples, device="mps", t_steps=1000):
    """
    Evolve x0 through the learned flow to produce a sample from p1.
    """
    xt = torch.randn(n_samples, 5, device=device)
    with torch.no_grad():
        for i, t_val in enumerate(torch.linspace(0, 1, t_steps, device=device), start=1):
            if i % 100 == 0:
                print(f"Generating samples. At time step {i}")
            pred = model(xt, t_val.expand(n_samples, 1))
            xt = xt + (1 / t_steps) * pred
    return xt


parser = argparse.ArgumentParser()

parser.add_argument("--correlations_file", type=str, default="vae_best.pth")
parser.add_argument("--plot_corr_post", action="store_true")
parser.add_argument("--plot_corr", action="store_true")
parser.add_argument("--loss", action="store_true")
parser.add_argument("--make_gif", action="store_true")
parser.add_argument("--n_samples", default=10_000)
args = parser.parse_args()
filename = args.correlations_file
batch_size = 1000
n_samples = int(args.n_samples)
device = "mps"


pr = cProfile.Profile()
pr.enable()
ckpt = torch.load("flowModel.pth", map_location=device)
loss = ckpt["loss"]
print(f"Loaded model showed a loss of {loss} on its batch")

nf = VelocityField(layers=5, network_dimensions=512).to(device)
nf.load_state_dict(ckpt["state_dict"])
nf.eval()
samples = sample_flow(nf, n_samples)
pr.disable()

lims = np.load("../prep_data_limits_nn.npy")
mins = torch.asarray(lims[0], dtype=torch.float32, device=device)
maxs = torch.asarray(lims[1], dtype=torch.float32, device=device)

mask = (samples >= mins) & (samples <= maxs)
samples = samples[mask.all(axis=1)].cpu().detach().numpy()
print(f"Actually plotted samples = {samples.shape[0]}")
if args.make_gif:
    print("Hello")


if args.plot_corr:
    plot_correlations_5d(samples, "Synthetic VAE: correlations preprocessed")
if args.plot_corr_post:
    samples = postprocess_nn(samples, 0, 0, filename="normalization_params.npy")
    plot_correlations_5d(samples, "Synthetic VAE: correlations postprocessed")

# Dump results:
# - for text dump
with open( 'cpu.txt', 'w') as output_file:
    sys.stdout = output_file
    pr.print_stats( sort='time' )
    sys.stdout = sys.__stdout__
plt.show()
