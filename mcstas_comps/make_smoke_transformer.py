"""Generates a synthetic gaussian_transformer_smoke.bin fixture for smoke-testing
Test_Source_ML.instr when no real preprocessed transformer
(../data_files/preprocess/gaussian_transformer.bin) is available. Not a
substitute for the real transform: values are unconstrained random Gaussians,
so downstream weights/positions/velocities are not physically meaningful.

Used only at export time (fed into export_onnx_smoke.py / export_torch_smoke.py,
which bake it into the exported model via
utils/data_load.py:InverseGaussRankTransform) -- Source_ML.comp itself never
reads a transform file at run time."""

import sys

import numpy as np

sys.path.append("../utils")
from data_load import save_transform_binary  # noqa: E402


def write_smoke_transformer(path="gaussian_transformer_smoke.bin"):
    rng = np.random.default_rng(0)
    n = 2000
    d = 12
    grid = (np.arange(n) + 0.5) / n
    sorted_cols = np.sort(rng.normal(size=(d, n)), axis=1)
    save_transform_binary(path, grid, sorted_cols)
    return path


if __name__ == "__main__":
    path = write_smoke_transformer()
    print(f"wrote {path}")
