"""PSD helpers and the three-way PSD comparison for icon_sample.instr.

Usage:
    python icon_psd_compare.py <ref_run_dir> <mcpl_run_dir> <ml_run_dir> <output_dir> [monitor ...]

The reference run uses an independent MCPL file. The MCPL run uses the original
MCPL file and the ML run the trained model. Each is compared to the reference.
"""
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_psd(run_dir, name):
    """Return (I, I_err, N, xylimits) from a McStas 2D monitor file."""
    with open(os.path.join(run_dir, name)) as f:
        lines = f.read().splitlines()
    shape = None
    xylim = None
    blocks = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# type: array_2d"):
            shape = [int(v) for v in line[line.index("(") + 1 : line.index(")")].split(",")]
        if line.startswith("# xylimits:"):
            xylim = [float(v) for v in line.split(":")[1].split()]
        if line.startswith(("# Data [", "# Errors [", "# Events [")):
            key = line.split()[1]
            rows = [[float(v) for v in lines[i + 1 + r].split()] for r in range(shape[1])]
            blocks[key] = np.array(rows)
            i += shape[1]
        i += 1
    return blocks["Data"], blocks["Errors"], blocks["Events"], xylim


def plot_image(img, xylim, title, filename, cmap="viridis", symmetric=False, label="I"):
    fig, ax = plt.subplots()
    vmax = np.abs(img).max() if symmetric else img.max()
    vmin = -vmax if symmetric else 0
    im = ax.imshow(img, origin="lower", extent=xylim, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(xlabel="x [cm]", ylabel="y [cm]", title=title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def compare(ref_dir, mcpl_dir, ml_dir, out_dir, monitor):
    """Save the PSDs of the three runs and the differences to the reference."""
    name = f"{monitor}.dat"
    I_r, E_r, _, xylim = load_psd(ref_dir, name)
    I_m, E_m, _, _ = load_psd(mcpl_dir, name)
    I_t, E_t, _, _ = load_psd(ml_dir, name)

    tag = monitor.replace("PSD_", "").replace("psd_", "")
    for label, run_dir, img in (("ref", ref_dir, I_r), ("mcpl", mcpl_dir, I_m), ("ml", ml_dir, I_t)):
        shutil.copy(os.path.join(run_dir, name), os.path.join(out_dir, f"ICON_PSD_{tag}_{label}.dat"))
        plot_image(img, xylim, f"{tag} PSD: {label}", os.path.join(out_dir, f"ICON_PSD_{tag}_{label}.png"))

    print(f"[{tag}] total I   reference: {I_r.sum():.6g}   mcpl: {I_m.sum():.6g}   ml: {I_t.sum():.6g}")
    for label, img, err in (("mcpl", I_m, E_m), ("ml", I_t, E_t)):
        diff = img - I_r
        np.savetxt(os.path.join(out_dir, f"ICON_PSD_{tag}_diff_{label}_minus_ref.dat"), diff)
        plot_image(
            diff, xylim, f"{tag} PSD difference: {label} - reference",
            os.path.join(out_dir, f"ICON_PSD_{tag}_diff_{label}_minus_ref.png"),
            cmap="RdBu_r", symmetric=True, label="dI",
        )
        total_err = np.sqrt(E_r**2 + err**2)
        mask = total_err > 0
        chi2 = np.sum((diff[mask] / total_err[mask]) ** 2) / mask.sum()
        print(f"[{tag}] {label} - ref: L2 {np.sum(diff**2):.6g}  chi2/pixel {chi2:.4g}  total I ratio {img.sum() / I_r.sum():.5g}")
        ny, nx = img.shape
        for iy, ix in [(ny // 2, nx // 2), (ny // 2, nx // 2 + 20), (ny // 2 + 20, nx // 2), (ny // 2 - 20, nx // 2 - 20)]:
            print(f"    pixel ({iy},{ix}): ref {I_r[iy, ix]:.5g}  {label} {img[iy, ix]:.5g}  err {total_err[iy, ix]:.3g}")


if __name__ == "__main__":
    ref_dir, mcpl_dir, ml_dir, out_dir = sys.argv[1:5]
    monitors = sys.argv[5:] or ["psd_focus", "psd_after_plate"]
    for monitor in monitors:
        compare(ref_dir, mcpl_dir, ml_dir, out_dir, monitor)
