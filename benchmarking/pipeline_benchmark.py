#!/usr/bin/env python
"""
End-to-end benchmark: for each generative model (CFM, VAE), train it, evaluate
it (which exports an ONNX and a TorchScript sampler), then run both the ONNX
(mcstas_comps/onnx_implementation) and TorchScript (mcstas_comps/torchscript)
McStas source components against the freshly exported model and report the
average time each backend takes to produce one batch of neutrons.

Usage (from anywhere, run inside the mcpl_torch environment or let
--env pick it up via `micromamba run`):

    python benchmarking/pipeline_benchmark.py
    python benchmarking/pipeline_benchmark.py --models cfm
    python benchmarking/pipeline_benchmark.py --n-particles 200000 --n-eval-samples 50000 --n-batches 5

Both models' train/eval scripts, and both mcstas instruments, are invoked as
subprocesses via `micromamba run -n <env>`, exactly as documented in
mcstas_comps/torchscript/README.md.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BATCH_SIZE = 10000  # SETTING PARAMETERS default in both Source_ML(.comp)/Source_ML_torch.comp

RUN_TIME_RE = re.compile(r"(?:ONNX Run|Torch Run): ([\d.]+) s")
GAUSS_INIT_RE = re.compile(r"Gaussian init: ([\d.]+) ms")
INVERSE_RE = re.compile(r"inverse transform: ([\d.]+) ms")


@dataclass
class ModelConfig:
    key: str
    display_name: str
    model_dir: str  # relative to REPO_ROOT
    train_script: str
    eval_script: str
    onnx_model_rel: str  # relative to mcstas_comps/*/ (i.e. from the instrument's cwd)
    torch_model_rel: str
    transformer_rel: str
    supports_n_particles: bool = True


MODEL_CONFIGS = {
    "cfm": ModelConfig(
        key="cfm",
        display_name="CFM",
        model_dir="models/CFM",
        train_script="train.py",
        eval_script="eval.py",
        onnx_model_rel="../../data_files/models/CFM_sampler.onnx",
        torch_model_rel="../../data_files/models/CFM_sampler.pt",
        transformer_rel="../../data_files/preprocess/gaussian_transformer.bin",
    ),
    "vae": ModelConfig(
        key="vae",
        display_name="VAE",
        model_dir="models/VAE",
        train_script="train.py",
        eval_script="eval.py",
        onnx_model_rel="../../data_files/models/VAE_sampler.onnx",
        torch_model_rel="../../data_files/models/VAE_sampler.pt",
        transformer_rel="../../data_files/preprocess/gaussian_transformer.bin",
    ),
}

ONNX_INSTR_DIR = "mcstas_comps/onnx_implementation"
ONNX_INSTR_FILE = "test.instr"
TORCH_INSTR_DIR = "mcstas_comps/torchscript"
TORCH_INSTR_FILE = "Test_Source_ML_torch.instr"


@dataclass
class BackendResult:
    backend: str
    n_batches: int
    first_batch_total_s: float
    steady_run_s: list = field(default_factory=list)
    steady_total_s: list = field(default_factory=list)

    @property
    def avg_run_s(self):
        return sum(self.steady_run_s) / len(self.steady_run_s)

    @property
    def avg_total_s(self):
        return sum(self.steady_total_s) / len(self.steady_total_s)

    @property
    def throughput_n_per_s(self):
        return BATCH_SIZE / self.avg_total_s


def log(msg):
    print(f"\n=== {msg} ===", flush=True)


def run_streamed(cmd, cwd, extra_env=None):
    """Run a subprocess, streaming its output live while also capturing it."""
    print(f"$ {' '.join(cmd)}   (cwd={cwd})", flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return "".join(lines)


def micromamba(env_name, args):
    return ["micromamba", "run", "-n", env_name] + args


def ensure_torchwrap_built(env_name, cc_override):
    """Build+install libtorchwrap into the active env if it isn't there yet,
    following mcstas_comps/torchscript/README.md."""
    prefix = subprocess.run(
        micromamba(env_name, ["printenv", "CONDA_PREFIX"]),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    lib_path = Path(prefix) / "lib" / "libtorchwrap.dylib"
    if lib_path.exists():
        return
    log("Building and installing libtorchwrap (not found in env)")
    torchscript_dir = REPO_ROOT / TORCH_INSTR_DIR
    torch_cmake = subprocess.run(
        micromamba(env_name, ["python", "-c", "import torch; print(torch.utils.cmake_prefix_path)"]),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    build_dir = torchscript_dir / "build"
    build_dir.mkdir(exist_ok=True)
    run_streamed(
        micromamba(env_name, [
            "cmake",
            f"-DCMAKE_PREFIX_PATH={torch_cmake}",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "..",
        ]),
        cwd=build_dir,
        extra_env={"CC": cc_override, "CXX": cc_override.replace("clang", "clang++")},
    )
    run_streamed(micromamba(env_name, ["make", "-j4"]), cwd=build_dir)
    run_streamed(micromamba(env_name, ["cmake", "--install", "."]), cwd=build_dir)


def train_model(cfg: ModelConfig, args):
    if args.skip_train:
        log(f"Skipping training for {cfg.display_name} (--skip-train)")
        return
    log(f"Training {cfg.display_name}")
    cmd = ["python", cfg.train_script, "--device", args.device]
    if args.n_particles is not None:
        cmd += ["--n_particles", str(int(args.n_particles))]
    run_streamed(micromamba(args.env, cmd), cwd=REPO_ROOT / cfg.model_dir)


def eval_model(cfg: ModelConfig, args):
    if args.skip_eval:
        log(f"Skipping eval for {cfg.display_name} (--skip-eval)")
        return
    log(f"Evaluating {cfg.display_name} (exports ONNX + TorchScript)")
    cmd = ["python", cfg.eval_script, "--device", args.device, "--n_samples", str(args.n_eval_samples)]
    run_streamed(
        micromamba(args.env, cmd),
        cwd=REPO_ROOT / cfg.model_dir,
        extra_env={"MPLBACKEND": "Agg"},
    )


def benchmark_component(backend, instr_dir, instr_file, params, args):
    log(f"Running {backend} McStas component ({args.n_batches} batches x {BATCH_SIZE} neutrons)")
    n_neutrons = args.n_batches * BATCH_SIZE
    param_args = [f"{k}={v}" for k, v in params.items()]
    cmd = micromamba(args.env, ["mcrun", "-n", str(n_neutrons), "--no-output-files", instr_file] + param_args)
    output = run_streamed(
        cmd,
        cwd=REPO_ROOT / instr_dir,
        extra_env={"MCSTAS_CC_OVERRIDE": args.cc_override},
    )

    run_times = [float(x) for x in RUN_TIME_RE.findall(output)]
    gauss_times = [float(x) / 1000.0 for x in GAUSS_INIT_RE.findall(output)]
    inverse_times = [float(x) / 1000.0 for x in INVERSE_RE.findall(output)]
    if not run_times:
        raise RuntimeError(f"No '{backend}' batch-timing lines found in mcrun output; check the run log above.")

    total_times = [g + r + i for g, r, i in zip(gauss_times, run_times, inverse_times)]

    result = BackendResult(
        backend=backend,
        n_batches=len(run_times),
        first_batch_total_s=total_times[0],
        steady_run_s=run_times[1:] or run_times,
        steady_total_s=total_times[1:] or total_times,
    )
    return result


def benchmark_model(cfg: ModelConfig, args):
    train_model(cfg, args)
    eval_model(cfg, args)

    onnx_result = benchmark_component(
        "ONNX",
        ONNX_INSTR_DIR,
        ONNX_INSTR_FILE,
        {"model_filename": cfg.onnx_model_rel},
        args,
    )
    torch_result = benchmark_component(
        "TorchScript",
        TORCH_INSTR_DIR,
        TORCH_INSTR_FILE,
        {"model_filename": cfg.torch_model_rel, "transformer_filename": cfg.transformer_rel},
        args,
    )
    return {"onnx": onnx_result, "torchscript": torch_result}


def print_summary(all_results):
    log("Summary: average time per neutron batch (steady-state, first batch excluded as warm-up)")
    header = f"{'model':<6} {'backend':<12} {'batches':>7} {'avg model-run (s)':>18} {'avg total/batch (s)':>20} {'neutrons/s':>12}"
    print(header)
    print("-" * len(header))
    for model_key, backends in all_results.items():
        for backend_key, r in backends.items():
            print(
                f"{model_key:<6} {r.backend:<12} {r.n_batches:>7} "
                f"{r.avg_run_s:>18.4f} {r.avg_total_s:>20.4f} {r.throughput_n_per_s:>12.0f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS), default=list(MODEL_CONFIGS),
                         help="Which model(s) to run the full train->eval->benchmark loop for.")
    parser.add_argument("--env", default="mcpl_torch", help="micromamba environment name.")
    parser.add_argument("--cc-override", default="/usr/bin/clang",
                         help="Value for MCSTAS_CC_OVERRIDE / CC when building the torch wrapper.")
    parser.add_argument("--device", default="mps", help="Device passed to train.py/eval.py.")
    parser.add_argument("--n-particles", type=float, default=None,
                         help="Override --n_particles passed to each model's train.py (default: script's own default).")
    parser.add_argument("--n-eval-samples", type=int, default=200_000,
                         help="--n_samples passed to each model's eval.py.")
    parser.add_argument("--n-batches", type=int, default=10,
                         help="Number of inference batches to time per mcstas component "
                              f"(neutrons simulated = n_batches * {BATCH_SIZE}).")
    parser.add_argument("--skip-train", action="store_true", help="Reuse the existing checkpoint instead of retraining.")
    parser.add_argument("--skip-eval", action="store_true", help="Reuse the existing exported models instead of re-evaluating.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optionally dump the summary results as JSON to this path.")
    args = parser.parse_args()

    ensure_torchwrap_built(args.env, args.cc_override)

    all_results = {}
    for key in args.models:
        cfg = MODEL_CONFIGS[key]
        log(f"##### {cfg.display_name} pipeline #####")
        all_results[key] = benchmark_model(cfg, args)

    print_summary(all_results)

    if args.json_out:
        serializable = {
            model: {
                backend: {
                    "n_batches": r.n_batches,
                    "first_batch_total_s": r.first_batch_total_s,
                    "avg_run_s": r.avg_run_s,
                    "avg_total_s": r.avg_total_s,
                    "throughput_n_per_s": r.throughput_n_per_s,
                }
                for backend, r in backends.items()
            }
            for model, backends in all_results.items()
        }
        args.json_out.write_text(json.dumps(serializable, indent=2))
        print(f"\nWrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
