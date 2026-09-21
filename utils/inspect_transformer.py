"""Inspect the inverse Gaussian-rank transform used by Source_ML.

Since #15, the transform is embedded directly in exported ONNX/TorchScript
models (utils/data_load.py:InverseGaussRankTransform) instead of living in a
separate gaussian_transformer.bin passed alongside the model. This script
extracts the transform's rank -> value lookup grid from any of the three
places it can live -- an .onnx model, a .pt/.pth TorchScript model, or a
standalone .bin file -- and plots the empirical inverse-CDF curve (quantile
on the x-axis, real-space value on the y-axis) for each dimension, so a
degenerate or heavy-tailed dimension is easy to spot by eye.

Usage:
    python inspect_transformer.py --model ../../data_files/models/CFM_sampler.onnx
    python inspect_transformer.py --model ../../data_files/models/CFM_sampler.pt \
        --dims weight energy "position x"
    python inspect_transformer.py --model ../../data_files/preprocess/gaussian_transformer.bin --show
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from data_load import load_transform_binary

DEFAULT_LABELS_12D = [
    "Weight [a.u]",
    "Energy [MeV]",
    "Time [ms]",
    "Direction x",
    "Direction y",
    "Direction z",
    "Position x [cm]",
    "Position y [cm]",
    "Position z [cm]",
    "Polarization x",
    "Polarization y",
    "Polarization z",
]


def _labels_for(d):
    if d == len(DEFAULT_LABELS_12D):
        return list(DEFAULT_LABELS_12D)
    return [f"dim {i}" for i in range(d)]


def load_from_onnx(path):
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    sorted_cols_t = None
    for init in model.graph.initializer:
        if init.name.endswith("sorted_cols_t"):
            sorted_cols_t = numpy_helper.to_array(init).astype(np.float64)
            break
    if sorted_cols_t is None:
        raise ValueError(
            f"No embedded gauss-rank transform found in '{path}' (expected "
            "an initializer named '*.sorted_cols_t'). Was this model "
            "exported with transformer_file_path set in "
            "export_model_as_onnx?"
        )
    # grid is always the fixed uniform quantile grid transform() builds; it
    # doesn't survive ONNX export as a named tensor (only its length is used,
    # via .shape[0], which the exporter constant-folds away), so rebuild it
    # from n instead of trying to recover it from the graph.
    n = sorted_cols_t.shape[0]
    grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    return grid, sorted_cols_t


def load_from_torchscript(path):
    import torch

    model = torch.jit.load(path, map_location="cpu")
    grid = sorted_cols_t = None
    for name, buf in model.named_buffers():
        if name.endswith("sorted_cols_t"):
            sorted_cols_t = buf.detach().cpu().numpy().astype(np.float64)
        elif name.endswith("grid"):
            grid = buf.detach().cpu().numpy().astype(np.float64)
    if sorted_cols_t is None:
        raise ValueError(
            f"No embedded gauss-rank transform found in '{path}' (expected "
            "buffers named '*.grid' / '*.sorted_cols_t'). Was this model "
            "exported with transformer_file_path set in "
            "export_model_as_torchscript?"
        )
    if grid is None:
        n = sorted_cols_t.shape[0]
        grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    return grid, sorted_cols_t


def load_from_bin(path):
    grid, sorted_cols = load_transform_binary(path)
    # load_transform_binary returns sorted_cols as (d, n); the rest of this
    # script uses (n, d), matching the model-buffer convention.
    return grid.astype(np.float64), sorted_cols.astype(np.float64).T


def load_transformer(path):
    """Returns (grid, sorted_cols_t): grid is (n,), sorted_cols_t is (n, d)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        return load_from_onnx(path)
    elif ext in (".pt", ".pth"):
        return load_from_torchscript(path)
    elif ext == ".bin":
        return load_from_bin(path)
    raise ValueError(
        f"Unrecognized file extension '{ext}' for '{path}'; expected "
        ".onnx, .pt/.pth, or .bin"
    )


def resolve_dims(dims, labels, d):
    if dims is None:
        return list(range(d))

    resolved = []
    lower_labels = [label.lower() for label in labels]
    for token in dims:
        try:
            idx = int(token)
        except ValueError:
            matches = [i for i, label in enumerate(lower_labels) if token.lower() in label]
            if not matches:
                raise ValueError(
                    f"'{token}' doesn't match any dimension label {labels} and isn't an integer index"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"'{token}' matches multiple dimension labels "
                    f"{[labels[i] for i in matches]}; be more specific"
                )
            idx = matches[0]
        else:
            if not (0 <= idx < d):
                raise ValueError(f"Dimension index {idx} out of range for a {d}-dimensional transform")
        resolved.append(idx)
    return resolved


def summarize(grid, sorted_cols_t, dim_indices, labels):
    print(f"{'dim':<20} {'min':>12} {'p25':>12} {'median':>12} {'p75':>12} {'max':>12}")
    for j in dim_indices:
        col = sorted_cols_t[:, j]
        label = labels[j] if j < len(labels) else f"dim {j}"
        p25, median, p75 = np.percentile(col, [25, 50, 75])
        print(f"{label:<20} {col.min():>12.5g} {p25:>12.5g} {median:>12.5g} {p75:>12.5g} {col.max():>12.5g}")


def plot_transformer(grid, sorted_cols_t, dim_indices, labels, title="", filename=""):
    ncols = min(4, len(dim_indices))
    nrows = int(np.ceil(len(dim_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, j in zip(axes.flat, dim_indices):
        label = labels[j] if j < len(labels) else f"dim {j}"
        ax.plot(grid, sorted_cols_t[:, j], lw=1)
        ax.set(xlabel="Quantile (rank / n)", ylabel=label, title=label)

    for ax in axes.flat[len(dim_indices):]:
        ax.set_axis_off()

    fig.suptitle(title or "Gauss-rank transform: rank -> value mapping")
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=200)
        print(f"Saved plot to {filename}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .onnx, .pt/.pth, or .bin transformer source")
    parser.add_argument(
        "--dims",
        nargs="+",
        default=None,
        help="Dimensions to inspect: integer indices and/or (partial, case-insensitive) label matches. Defaults to all.",
    )
    parser.add_argument("--filename", default="", help="Path to save the plot to. Defaults to <model_stem>_transformer.png")
    parser.add_argument("--title", default="", help="Plot title override")
    parser.add_argument("--show", action="store_true", help="Also display the plot interactively")
    args = parser.parse_args()

    grid, sorted_cols_t = load_transformer(args.model)
    n, d = sorted_cols_t.shape
    labels = _labels_for(d)
    dim_indices = resolve_dims(args.dims, labels, d)

    print(f"Loaded transform from '{args.model}': {n} grid points, {d} dimensions")
    summarize(grid, sorted_cols_t, dim_indices, labels)

    filename = args.filename or f"{os.path.splitext(os.path.basename(args.model))[0]}_transformer.png"
    plot_transformer(grid, sorted_cols_t, dim_indices, labels, title=args.title, filename=filename)

    if args.show:
        plt.show()
