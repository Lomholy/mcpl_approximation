"""Generates a synthetic gaussian_transformer_smoke.bin fixture for smoke-testing
Test_Source_ML_torch.instr when no real preprocessed transformer
(../../data_files/preprocess/gaussian_transformer.bin) is available. Not a
substitute for the real transform: values are unconstrained random Gaussians,
so downstream weights/positions/velocities are not physically meaningful."""

import sys

import numpy as np

sys.path.append("../../utils")
from data_load import save_transform_binary  # noqa: E402

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    d = 12
    grid = (np.arange(n) + 0.5) / n
    sorted_cols = np.sort(rng.normal(size=(d, n)), axis=1)
    save_transform_binary("gaussian_transformer_smoke.bin", grid, sorted_cols)
    print("wrote gaussian_transformer_smoke.bin")
