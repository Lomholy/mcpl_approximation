# mcpl_approximation
A repository containing my attempt at using machine learning methods to approximate the underlying distribution of a Monte Carlo Particle List (MCPL) file.

Two generative models (a Conditional Flow Matching model and a VAE, under
`models/CFM` and `models/VAE`) are trained on a real MCPL neutron event file
so they can later be sampled from as a drop-in replacement for that file.
Once trained, each model is exported to ONNX and TorchScript and wired into
a McStas source component (`mcstas_comps/onnx_implementation` and
`mcstas_comps/torchscript`) so it can be sampled from directly inside a
McStas instrument, instead of reading events from a static `.mcpl` file.

## Repository layout

- `models/CFM`, `models/VAE` — model definitions (`model.py`) and
  `train.py`/`eval.py` scripts. `eval.py` also exports the trained model to
  ONNX and TorchScript via `utils/data_load.py`.
- `utils/` — shared helpers: loading/writing MCPL files (`data_load.py`),
  plotting correlations (`plotting.py`, `plot_model_correlations.py`,
  `plot_mcpl.py`), and reference distributions (`true_correlations.py`).
- `mcstas_comps/onnx_implementation` — the ONNX-backed `Source_ML` McStas
  component and its C/Python test harness.
- `mcstas_comps/torchscript` — the TorchScript-backed `Source_ML_torch`
  McStas component (a C++ LibTorch bridge plus the component itself); see
  [its README](mcstas_comps/torchscript/README.md) for the build/run steps.
- `benchmarking/` — `pipeline_benchmark.py` runs the full
  train → eval/export → McStas end-to-end for both models and both backends
  and reports timings; `benchmarking/L2` holds scripts for splitting an
  MCPL file into subsamples and comparing McStas run outputs against them.
- `data_files/` — inputs and generated artifacts (MCPL files, trained
  model exports, losses, samples). Populated locally; not committed.

## Environment setup

All scripts (training, evaluation/export, and the McStas component builds
and runs) are designed to run inside a single conda/mamba environment,
described in [`environment.yml`](environment.yml). Create and activate it
with:

```bash
mamba env create -f environment.yml
mamba activate mcpl_torch
```

This installs Python 3.11, CPU PyTorch, ONNX/ONNX Runtime (including the
C API used by `mcstas_comps/onnx_implementation`), McStas and
McStasScript, MCPL's Python bindings, and a C/C++ compiler toolchain plus
CMake for building the TorchScript bridge in `mcstas_comps/torchscript`.

If you only need to build/run the TorchScript component, see
[`mcstas_comps/torchscript/README.md`](mcstas_comps/torchscript/README.md)
for the extra one-time build step (compiling `libtorchwrap` with CMake)
and known macOS compiler-toolchain caveats.

## Basic usage

With the environment active, train and evaluate a model, e.g. the CFM:

```bash
python models/CFM/train.py
python models/CFM/eval.py   # also exports CFM_sampler.onnx / CFM_sampler.pt
```

`eval.py` writes its ONNX/TorchScript exports into `data_files/models/`,
which the McStas components in `mcstas_comps/onnx_implementation` and
`mcstas_comps/torchscript` read from when run as instrument sources.

To run the whole pipeline (train, eval/export, and both McStas backends)
for both models and get a timing comparison, use:

```bash
python benchmarking/pipeline_benchmark.py
```

## Author:
Daniel Lomholt Christensen
