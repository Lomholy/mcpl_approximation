# mcpl_approximation
A repository containing my attempt at using machine learning methods to approximate the underlying distribution of a Monte Carlo Particle List (MCPL) file.

Two generative models (a Conditional Flow Matching model and a VAE, under
`models/CFM` and `models/VAE`) are trained on a real MCPL neutron event file
so they can later be sampled from as a drop-in replacement for that file.
Once trained, each model is exported to ONNX and TorchScript (with the
inverse Gaussian-rank preprocessing transform baked into the exported graph
itself) and wired into `Source_ML`, a single McStas source component
(`mcstas_comps/Source_ML`) that dispatches between the two backends at
runtime, so the model can be sampled from directly inside a McStas
instrument, instead of reading events from a static `.mcpl` file.

## Repository layout

- `models/CFM`, `models/VAE` — model definitions (`model.py`) and
  `train.py`/`eval.py` scripts. `eval.py` also exports the trained model to
  ONNX and TorchScript via `utils/data_load.py`.
- `utils/` — shared helpers: loading/writing MCPL files (`data_load.py`),
  plotting correlations (`plotting.py`, `plot_model_correlations.py`,
  `plot_mcpl.py`), and reference distributions (`true_correlations.py`).
- `mcstas_comps/Source_ML` — the `Source_ML` McStas component (backed by
  `libmlbackend`, a C++ library supporting both ONNX Runtime and, optionally,
  LibTorch) and its C/Python test harness; see
  [its README](mcstas_comps/Source_ML/README.md) for the build/run steps.
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

This installs Python 3.11, CPU PyTorch, ONNX/ONNX Runtime (including
`onnxruntime-cpp`'s C API, the one dependency `mcstas_comps/Source_ML`'s
library, `libmlbackend`, requires unconditionally), McStas and
McStasScript, MCPL's Python bindings, and a C/C++ compiler toolchain plus
CMake for building `libmlbackend`. PyTorch/LibTorch is optional for
`libmlbackend` itself -- it's auto-detected at build time, and a build
without it still works fully for ONNX models (see
[`mcstas_comps/Source_ML/README.md`](mcstas_comps/Source_ML/README.md)).

Before running an instrument, `libmlbackend` needs a one-time CMake build
and install step -- see
[`mcstas_comps/Source_ML/README.md`](mcstas_comps/Source_ML/README.md) for
that and known macOS compiler-toolchain caveats.

## Basic usage

With the environment active, train and evaluate a model, e.g. the CFM:

```bash
python models/CFM/train.py
python models/CFM/eval.py   # also exports CFM_sampler.onnx / CFM_sampler.pt
```

`eval.py` writes its ONNX/TorchScript exports into `data_files/models/`,
which `mcstas_comps/Source_ML`'s `Source_ML` component reads from when run
as an instrument source (picking the ONNX or TorchScript backend based on
the file extension, or an explicit `backend=` parameter).

To run the whole pipeline (train, eval/export, and both McStas backends)
for both models and get a timing comparison, use:

```bash
python benchmarking/pipeline_benchmark.py
```

## Author:
Daniel Lomholt Christensen
