import sys
sys.path.append("../../utils/")
from data_load import inverse_transform, get_transformer_limits, export_model_as_onnx, export_model_as_torchscript
from plotting import plot_correlations_12d
from model import VelocityField, Sampler
import torch
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("--plot", action="store_true")
parser.add_argument("--loss", action="store_true")
parser.add_argument("--n_samples", default=1_000_000)
args = parser.parse_args()
batch_size = 10000
n_samples = int(args.n_samples)
device = "mps"


ckpt = torch.load("../../data_files/models/CFM.pth", map_location=device)

model = VelocityField().to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

lims, grid, cols = get_transformer_limits(file_path="../../data_files/preprocess/gaussian_transformer.bin")

mins = torch.asarray(lims[0], dtype=torch.float32, device="cpu")
maxs = torch.asarray(lims[1], dtype=torch.float32, device="cpu")


samples = []
gaussian_samples = []
sampler = Sampler(model, device=device)
while len(samples) < n_samples:
    start = time.time()
    print(len(samples))
    # Limit samples to within the preprocessed data limits
    x = torch.randn(batch_size, 12, device=device)
    batch = sampler.forward(x).cpu().detach().numpy()
    gaussian_batch = batch
    batch = torch.asarray(inverse_transform(batch, file_path="../../data_files/preprocess/gaussian_transformer.bin"))
    mask = (batch >= mins) & (batch <= maxs)
    batch = batch[mask.all(axis=1)]
    gaussian_samples = gaussian_samples + gaussian_batch[mask.all(axis=1)].tolist()
    samples = samples + batch.tolist()
    print(f"Iteration time = {time.time() - start}")
print(next(sampler.parameters()).dtype)
export_model_as_onnx(sampler, "../../data_files/models/CFM_sampler.onnx", device)
export_model_as_torchscript(sampler, "../../data_files/models/CFM_sampler.pt", device)


samples = samples[:n_samples]
gaussian_samples = gaussian_samples[:n_samples]
samples = torch.asarray(samples)
gaussian_samples = torch.asarray(gaussian_samples)
torch.save(gaussian_samples, "../../data_files/samples/CFM_gauss.pkl")
torch.save(samples, "../../data_files/samples/CFM.pkl")



print(f"Actually plotted samples = {samples.shape[0]}")
if args.plot:
    plot_correlations_12d(gaussian_samples, "CFM: Gaussian space", filename="../../figures/CFM_gauss.png")
    plot_correlations_12d(samples, "CFM: neutron phase space", filename="../../figures/CFM_neutron.png")

# torch.save(gaussian_samples, "../../data_files/model_samples/cfm_gauss.pkl")
plt.show()
