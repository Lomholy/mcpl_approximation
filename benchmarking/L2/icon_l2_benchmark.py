"""L2 benchmark of the PSI_ICON MCPL file and a trained model against a reference MCPL file.

The reference is a second MCPL file recorded at the same plane with an
independent random seed. The sample instrument (icon_resolution.instr by
default, a mask with sharp edges and small features, or icon_sample.instr, a
zone plate) is run with the reference to give a reference PSD. For each
particle count n:

- MCPL side: the instrument runs on a random subset of n neutrons of the
  original MCPL file, with weights rescaled by N_total / n.
- ML side: the Source_ML component runs directly with ncount=n, so no MCPL file
  is written for the model.

The loss is sum((I_ref - I)^2) over the PSD. The n = N_total point of the MCPL
side is the whole original file.

Usage (from benchmarking/L2, with MCSTAS_CC_OVERRIDE and CONDA_PREFIX set as in
mcstas_comps/README.md):
    python icon_l2_benchmark.py --ref-mcpl <reference mcpl> --out-dir ~/Desktop

For the zone plate:
    python icon_l2_benchmark.py ... --instrument icon_sample.instr --monitors psd_focus psd_after_plate
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import np2mcpl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.append(str(REPO / "utils"))
sys.path.append(str(HERE))
from data_load import load_mcpl_file_random  # noqa: E402
from icon_psd_compare import compare, load_psd  # noqa: E402

def make_ml_instr(src, dst):
    """Write a copy of the sample instrument without the MCPL source, whose
    INITIALIZE would otherwise override ncount."""
    text = Path(src).read_text()
    text = re.sub(r"COMPONENT MCPL_source = MCPL_input\(.*?ABSOLUTE\n\n", "", text, flags=re.S)
    text = re.sub(r'string run_from_mcpl = ".*?",\n', "", text)
    text = re.sub(r"int use_ml = 0,\n", "", text)
    text = text.replace(" WHEN (use_ml)", "")
    text = text.replace(Path(src).stem, Path(src).stem + "_ml")
    Path(dst).write_text(text)


def write_mcpl(data, name):
    out = np.zeros((len(data), 10), dtype=np.float32)
    out[:, 0] = 2112
    out[:, 1:4] = data[:, 6:9]
    out[:, 4:7] = data[:, 3:6]
    out[:, 7] = data[:, 2]
    out[:, 8] = data[:, 1]
    out[:, 9] = data[:, 0]
    np2mcpl.save(name, out)


def build(instr, work, params):
    """Compile instr in work. mcrun also runs the binary once, which only
    needs to not wait on stdin."""
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "mcstas_comps" / "Source_ML.comp", work / "Source_ML.comp")
    shutil.copy(instr, work / instr.name)
    cmd = ["mcrun", "-c", instr.name, "-n", "10", "-d", "build_run"] + params
    subprocess.run(cmd, cwd=work, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    binary = work / (instr.stem + ".out")
    if not binary.exists():
        raise RuntimeError(f"Compilation of {instr} failed")
    return binary


def run(binary, n, outdir, params):
    if Path(outdir).exists():
        shutil.rmtree(outdir)
    t0 = time.time()
    subprocess.run(
        [str(binary), "-n", str(int(n)), "-d", str(outdir)] + params,
        cwd=HERE, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    return time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-mcpl", required=True)
    ap.add_argument("--instrument", default="icon_resolution.instr",
                    help="Sample instrument in this directory")
    ap.add_argument("--monitors", nargs="+", default=["psd_flush", "psd_far"],
                    help="PSD monitors of the instrument to compare")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", default=str(HERE / "icon_work"))
    ap.add_argument("--mcpl", default=str(REPO / "data_files/mcpl_files/PSI_ICON.mcpl.gz"))
    ap.add_argument("--model", default=str(REPO / "data_files/models/CFM_sampler.pt"))
    ap.add_argument("--n-train", type=int, default=1_000_000)
    ap.add_argument("--n-sizes", type=int, default=14)
    ap.add_argument("--max-size", type=int, default=30_000_000)
    ap.add_argument("--skip-sizes-above", type=int, default=None,
                    help="Skip subset sizes above this value (the full file is still run)")
    args = ap.parse_args()
    MONITORS = args.monitors
    instrument = HERE / args.instrument

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)

    full = load_mcpl_file_random(args.mcpl, 10**9).numpy()
    n_total = len(full)
    print(f"Loaded {n_total} neutrons", flush=True)

    write_mcpl(full[:2000], str(work / "build_small"))
    mcpl_bin = build(instrument, work / "mcpl_build",
                     [f"run_from_mcpl={work}/build_small.mcpl.gz"])
    ml_instr = work / f"{instrument.stem}_ml.instr"
    make_ml_instr(instrument, ml_instr)
    ml_params = [f"run_ml={args.model}"]
    ml_bin = build(ml_instr, work / "ml_build", ml_params)

    ref_dir = work / "run_ref"
    print("reference run", run(mcpl_bin, 1, ref_dir, [f"run_from_mcpl={args.ref_mcpl}"]), flush=True)
    ref = {m: load_psd(ref_dir, f"{m}.dat")[0] for m in MONITORS}

    sizes = np.unique(np.geomspace(1_000, args.max_size, args.n_sizes).astype(int))
    if args.skip_sizes_above is not None:
        sizes = sizes[sizes <= args.skip_sizes_above]

    rows_file = work / "rows.json"
    rows = json.loads(rows_file.read_text()) if rows_file.exists() else []
    done = {r["n"] for r in rows}

    for n in list(sizes) + [n_total]:
        if int(n) in done:
            continue
        rng = np.random.default_rng(int(n))
        row = {"n": int(n)}

        if n < n_total:
            idx = rng.choice(n_total, n, replace=False)
            sub = full[idx].copy()
            sub[:, 0] *= n_total / n
            name = work / f"subset_{n}"
            write_mcpl(sub, str(name))
            mcpl_file = f"{name}.mcpl.gz"
            d = work / f"run_mcpl_{n}"
        else:
            mcpl_file = args.mcpl
            d = work / "run_mcpl_full"
        row["t_mcpl"] = run(mcpl_bin, 1, d, [f"run_from_mcpl={mcpl_file}"])
        for m in MONITORS:
            I = load_psd(d, f"{m}.dat")[0]
            row[f"I_mcpl_{m}"] = float(I.sum())
            row[f"loss_mcpl_{m}"] = float(np.sum((ref[m] - I) ** 2))
        if n < n_total:
            os.remove(mcpl_file)
            shutil.rmtree(d)

        d = work / (f"run_ml_{n}" if n < n_total else "run_ml_full")
        row["t_ml"] = run(ml_bin, n, d, ml_params)
        for m in MONITORS:
            I = load_psd(d, f"{m}.dat")[0]
            row[f"I_ml_{m}"] = float(I.sum())
            row[f"loss_ml_{m}"] = float(np.sum((ref[m] - I) ** 2))
        if n < n_total:
            shutil.rmtree(d)

        rows.append(row)
        rows_file.write_text(json.dumps(rows))
        print(row, flush=True)

    rows.sort(key=lambda r: r["n"])

    with open(out_dir / "ICON_L2_loss_table.csv", "w") as f:
        f.write("n," + ",".join(f"I_ref_{m},I_mcpl_{m},I_ml_{m},loss_mcpl_{m},loss_ml_{m}" for m in MONITORS) + "\n")
        for r in rows:
            f.write(
                f"{r['n']},"
                + ",".join(
                    f"{ref[m].sum():.6g},{r[f'I_mcpl_{m}']:.6g},{r[f'I_ml_{m}']:.6g},"
                    f"{r[f'loss_mcpl_{m}']:.6g},{r[f'loss_ml_{m}']:.6g}"
                    for m in MONITORS
                )
                + "\n"
            )

    fig, axes = plt.subplots(1, len(MONITORS), figsize=(6 * len(MONITORS), 4.8), squeeze=False)
    for ax, m in zip(axes[0], MONITORS):
        ax.plot([r["n"] for r in rows], [r[f"loss_mcpl_{m}"] for r in rows], "+b", label="Original MCPL subset (MCPL_input)")
        ax.plot([r["n"] for r in rows], [r[f"loss_ml_{m}"] for r in rows], "+r", label="CFM (Source_ML)")
        ax.axvline(args.n_train, color="purple", linestyle="--", label=f"Trained on {args.n_train:.0e} neutrons")
        ax.axvline(n_total, color="gray", linestyle=":", label=f"Full original file ({n_total:.3g} neutrons)")
        ax.set(xscale="log", yscale="log", xlabel="Number of particles [#]",
               ylabel="L2 loss to reference MCPL [sum dI**2]", title=m)
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ICON_L2_loss_comparison.png", dpi=150)

    for m in MONITORS:
        compare(ref_dir, work / "run_mcpl_full", work / "run_ml_full", out_dir, m)
