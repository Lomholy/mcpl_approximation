import sys
sys.path.append("..")
from plotting import plot_correlations_7d
from data_load import postprocess_nn, save_data_as_mcpl
from nf_definition import VelocityField
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import time

def sample_flow(model, n_samples, device="mps", t_steps=256):
    """
    Evolve x0 through the learned flow to produce a sample from p1.
    """
    xt = torch.randn(n_samples, 7, device=device)
    with torch.no_grad():
        for i, t_val in tqdm(enumerate(torch.linspace(0, 1, t_steps, device=device), start=1)):
            pred = model(xt, t_val.expand(n_samples, 1))
            xt = xt + (1 / t_steps) * pred
    return xt


parser = argparse.ArgumentParser()

parser.add_argument("--plot", action="store_true")
parser.add_argument("--loss", action="store_true")
parser.add_argument("--n_samples", default=1_000_000)
args = parser.parse_args()
batch_size = 10000
n_samples = int(args.n_samples)
device = "mps"


ckpt = torch.load("flowModel.pth", map_location=device)

nf = VelocityField().to(device)
nf.load_state_dict(ckpt["state_dict"])
nf.eval()

lims = np.load("../limits.npy") 


mins = torch.asarray(lims[0], dtype=torch.float32, device="cpu")
maxs = torch.asarray(lims[1], dtype=torch.float32, device="cpu")
with open('gaussian_transformer.pkl', 'rb') as inp:
    transformer = pickle.load(inp)


samples = []
gaussian_samples = []
start = time.time()
while len(samples) < n_samples:
    print(len(samples))
    # Limit samples to within the preprocessed data limits
    batch = sample_flow(nf, batch_size).cpu()
    gaussian_samples = gaussian_samples + batch.tolist()
    batch = torch.asarray(transformer.inverse_transform(batch))
    mask = (batch >= mins) & (batch <= maxs)
    batch = batch[mask.all(axis=1)]
    samples = samples + batch.tolist()
    print(f"Time passed = {time.time() - start}")
samples = samples[:n_samples]
gaussian_samples = gaussian_samples[:n_samples]
samples = torch.asarray(samples)
gaussian_samples = torch.asarray(gaussian_samples)

print(f"Actually plotted samples = {samples.shape[0]}")
if args.plot:
    plot_correlations_7d(gaussian_samples, "Synthetic CFM: Gaussian samples", filename="gauss_output.png")
    plot_correlations_7d(samples, "Synthetic CFM: correlations", filename="raw_output.png")

save_data_as_mcpl(samples, "../cmf_samples")
torch.save(gaussian_samples, "../gaussian_output.pkl")
plt.show()
