import joblib
import torch
from kde_definition import AdaptiveKDE
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from data_load import load_mcpl_file, preprocess, postprocess
from plotting import plot_correlations_6d

# ---------------------------------------------------------
# Global multivariate bandwidth matrix
# (Scott's rule for dimension d)
# ---------------------------------------------------------


def global_bandwidth_matrix(data):
    n, d = data.shape
    cov = torch.cov(data.T)
    factor = n ** (-1.0 / (d + 4))
    return cov * factor**2


# ---------------------------------------------------------
# Multivariate Gaussian kernel
# ---------------------------------------------------------
def gaussian_kernel(x, data, H_inv, H_det):
    diff = data - x  # [N, d]
    # q = diff^T H_inv diff
    q = torch.einsum("nd,dd,nd->n", diff, H_inv, diff)
    d = data.shape[1]
    norm = torch.sqrt((2 * torch.pi) ** d * H_det)
    return torch.exp(-0.5 * q) / norm


# ---------------------------------------------------------
# Pilot KDE (global bandwidth)
# ---------------------------------------------------------


def pilot_density(data, H, batch_size=5000):
    H_inv = torch.linalg.inv(H)
    H_det = torch.linalg.det(H)
    N = data.shape[0]

    pilot = torch.zeros(N, device=data.device)
    for i in range(0, N, batch_size):
        batch = data[i:i+batch_size]
        for j in range(0, N, batch_size):
            block = data[j:j+batch_size] # [B2, d]
            diff = batch[:, None, :] - block[None, :, :]  # [B, B2, d]

            q = torch.einsum("bij,dd,bij->bi", diff, H_inv, diff)
            vals = torch.exp(-0.5*q) / torch.sqrt((2*torch.pi)**data.shape[1] * H_det)

            pilot[i:i+batch_size] += vals.sum(dim=1)

    return pilot


def pilot_density_batched(data, H, batch_size=5000):
    H_inv = torch.linalg.inv(H)
    H_det = torch.linalg.det(H)
    N = data.shape[0]

    pilot = torch.zeros(N, device=data.device)

    for i in range(0, N, batch_size):
        batch = data[i:i+batch_size]     # shape [B, d]

        # diff is [B, N, d] → too big  
        # So compute kernel row-wise in smaller chunks:
        for j in range(0, N, batch_size):
            block = data[j:j+batch_size] # [B2, d]
            diff = batch[:, None, :] - block[None, :, :]  # [B, B2, d]

            q = torch.einsum("bij,dd,bij->bi", diff, H_inv, diff)
            vals = torch.exp(-0.5*q) / torch.sqrt((2*torch.pi)**data.shape[1] * H_det)

            pilot[i:i+batch_size] += vals.sum(dim=1)

    return pilot

# ---------------------------------------------------------
# Compute adaptive bandwidth factors (Abramson)
# ---------------------------------------------------------
def adaptive_bandwidths(pilot, epsilon=1e-10):
    g = torch.exp(torch.mean(torch.log(pilot + epsilon)))
    return torch.sqrt(g / (pilot + epsilon))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
device = "mps"
raw_data = load_mcpl_file("../ODIN.mcpl.gz", int(1e+6))

pre_data, _, _, _, _ = preprocess(raw_data)

# plot_correlations_6d(pre_data, "Input data kde")
pre_data = pre_data.to(device)

print("Computing global bandwidth matrix")
# Compute global bandwidth matrix
H = global_bandwidth_matrix(pre_data)

print("Computig pilot density")
# Compute pilot density
pilot = pilot_density(pre_data, H)

print("Computing local scaling factors")
# Compute local scaling factors
local_factors = adaptive_bandwidths(pilot)

local_factors = local_factors.cpu().detach().numpy()
print("Creating the kde object")
# Build adaptive KDE object
kde = AdaptiveKDE(
    pre_data.cpu().detach().numpy(), H.cpu().detach().numpy(), local_factors

)

joblib.dump(kde, "kde.pkl")
print("Saved model as kde.pkl")
plt.show()
