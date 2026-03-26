import joblib
import torch
from adaptive_kde import AdaptiveKDE
import sys
sys.path.append("..")
from data_load import load_mcpl_file, preprocess

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


def pilot_density(data, H):
    H_inv = torch.linalg.inv(H)
    H_det = torch.linalg.det(H)

    vals = []
    for i in range(data.shape[0]):
        vals.append(gaussian_kernel(data[i], data, H_inv, H_det).mean())
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
    data = load_mcpl_file("../ODIN.mcpl.gz", 10)
    data = preprocess(data)[0].to(device)

    print("Computing global bandwidth matrix")
    # Compute global bandwidth matrix
    H = global_bandwidth_matrix(data)

    print("Computig pilot density")
    # Compute pilot density
    pilot = pilot_density(data, H)

    print("Computing local scaling factors")
    # Compute local scaling factors
    local_factors = adaptive_bandwidths(pilot)

    local_factors = local_factors.cpu().detach().numpy()
    print("Creating the kde object")
    # Build adaptive KDE object
    kde = AdaptiveKDE(
        data.cpu().detach().numpy(), H.cpu().detach().numpy(), local_factors
    )

    joblib.dump(kde, "adaptive_kde.pkl")
    print("Saved model as kde_model.pkl")
