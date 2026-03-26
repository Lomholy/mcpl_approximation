import numpy as np
from numpy.linalg import det, inv
from scipy.spatial.distance import cdist
import torch


class AdaptiveKDE:
    def __init__(self, data, H, local_factors):
        self.data = data
        self.H = H
        self.local_factors = local_factors
        self.H_inv = inv(H)
        self.H_det = det(H)

    def evaluate(self, x):
        x = np.atleast_2d(x)
        vals = np.zeros(x.shape[0])

        for i, xi in enumerate(x):
            diffs = xi - self.data
            H_i_inv = (self.local_factors.reshape(-1, 1, 1) ** -2) * self.H_inv
            H_i_det = (self.local_factors**2) ** (self.data.shape[1]) * self.H_det

            q = np.einsum("nij,nj->ni", H_i_inv, diffs)
            q = np.sum(diffs * q, axis=1)

            kernels = np.exp(-0.5 * q) / np.sqrt(
                (2 * np.pi) ** self.data.shape[1] * H_i_det
            )
            vals[i] = np.mean(kernels)

        return vals

    def sample(self, n):
        idx = np.random.choice(len(self.data), size=n, replace=True)
        base = self.data[idx]
        d = self.data.shape[1]
        eps = np.random.randn(n, d)
        samples = np.zeros((n, d))

        for i in range(n):
            L = np.linalg.cholesky(self.H) * self.local_factors[idx][i]
            samples[i] = base[i] + eps[i] @ L.T
        return samples


class AdaptiveKDE_Torch:
    def __init__(self, data, H, local_factors):
        self.data = data
        self.H = H
        self.local_factors = local_factors
        self.H_inv = torch.linalg.inv(H)
        self.H_det = torch.linalg.det(H)

    def sample(self, n):
        idx = torch.randint(0, self.data.shape[0], (n,), device=self.data.device)
        base = self.data[idx]

        d = self.data.shape[1]
        eps = torch.randn(n, d, device=self.data.device)

        samples = torch.zeros_like(base)

        for i in range(n):
            L = torch.linalg.cholesky(self.H) * self.local_factors[idx[i]]
            samples[i] = base[i] + eps[i] @ L.T

        return samples.cpu().numpy()
