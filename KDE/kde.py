import joblib
import torch
from kde_definition import AdaptiveKDE
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from data_load import load_mcpl_file, preprocess
from plotting import plot_correlations_5d

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


def pilot_density(data, H, weights):
    H_inv = torch.linalg.inv(H)
    H_det = torch.linalg.det(H)

    vals = []
    for i in range(data.shape[0]):
        vals.append(torch.dot(gaussian_kernel(data[i], data, H_inv, H_det), weights))
    return torch.stack(vals)


# ---------------------------------------------------------
# Compute adaptive bandwidth factors (Abramson)
# ---------------------------------------------------------
def adaptive_bandwidths(pilot, epsilon=1e-10):
    g = torch.exp(torch.mean(torch.log(pilot + epsilon)))
    return torch.sqrt(g / (pilot + epsilon))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    device = "mps"
    raw_data = load_mcpl_file("../ODIN.mcpl.gz", 1e+5)
    pre_data, weights, _, _, _, _ = preprocess(raw_data)
    total_intensity = weights.sum()
    norm_weights = weights/total_intensity
    norm_weights = norm_weights.to(device)

    plot_correlations_5d(pre_data, "Input data kde")
    pre_data = pre_data.to(device)

    print("Computing global bandwidth matrix")
    # Compute global bandwidth matrix
    H = global_bandwidth_matrix(pre_data)

    print("Computig pilot density")
    # Compute pilot density
    pilot = pilot_density(pre_data, H, norm_weights)

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
