# mcpl_approximation
A repository containing my attempt at using machine learning methods to approximate the underlying distribution of a Monte Carlo Particle List (MCPL) file.

Two generative models (a Conditional Flow Matching model and a VAE, under
`models/CFM` and `models/VAE`) are trained on a real MCPL neutron event file
so they can later be sampled from as a drop-in replacement for that file.
Once trained, each model is exported to ONNX and TorchScript (with the
inverse Gaussian-rank preprocessing transform baked into the exported graph
itself) and wired into `Source_ML`, a single McStas source component (in
`mcstas_comps`) that dispatches between the two backends at runtime, so the
model can be sampled from directly inside a McStas instrument, instead of
reading events from a static `.mcpl` file.

## Repository layout

- `models/CFM`, `models/VAE` — model definitions (`model.py`) and
  `train.py`/`eval.py` scripts. `eval.py` also exports the trained model to
  ONNX and TorchScript via `utils/data_load.py`.
- `utils/` — shared helpers: loading/writing MCPL files (`data_load.py`),
  plotting correlations (`plotting.py`, `plot_model_correlations.py`,
  `plot_mcpl.py`), and reference distributions (`true_correlations.py`).
- `mcstas_comps` — the `Source_ML` McStas component (backed by
  `libmlbackend`, a C++ library supporting both ONNX Runtime and, optionally,
  LibTorch) and its C/Python test harness; see
  [its README](mcstas_comps/README.md) for the build/run steps.
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
`onnxruntime-cpp`'s C API, the one dependency `mcstas_comps/libmlbackend`
requires unconditionally), McStas and
McStasScript, MCPL's Python bindings, and a C/C++ compiler toolchain plus
CMake for building `libmlbackend`. PyTorch/LibTorch is optional for
`libmlbackend` itself -- it's auto-detected at build time, and a build
without it still works fully for ONNX models (see
[`mcstas_comps/README.md`](mcstas_comps/README.md)).

Before running an instrument, `libmlbackend` needs a one-time CMake build
and install step -- see step 2 of the guide below, and
[`mcstas_comps/README.md`](mcstas_comps/README.md) for the details and
known macOS compiler-toolchain caveats.

## Guide: from an MCPL file to a source in your instrument

This guide starts with an MCPL file and ends with a McStas instrument, in a
folder of your own, that samples neutrons from a model trained on that file.
It works for both models, CFM and VAE. All commands assume a fresh clone of
this repository.

### 1. Set up the environment (once)

```bash
mamba env create -f environment.yml
mamba activate mcpl_torch
```

Keep this environment active for every step below, including when you run
your instrument.

### 2. Build the component's library (once)

`Source_ML` calls into a small library, `libmlbackend`, that has to be
installed into the environment:

```bash
cd mcstas_comps
make install
cd ..
```

On macOS, if this fails with linker errors (`ld: ... malformed file`), build
with the system compiler instead:

```bash
CC=/usr/bin/clang CXX=/usr/bin/clang++ make install
```

Check that it is installed:

```bash
ls $CONDA_PREFIX/lib/libmlbackend.* $CONDA_PREFIX/include/ml_backend.h
```

### 3. Put your MCPL file in place

The `data_files` folders are not part of the repository and the scripts do
not create them, so make them first, from the repository root:

```bash
mkdir -p data_files/{mcpl_files,preprocess,samples,models,losses} figures
cp /path/to/my_source.mcpl.gz data_files/mcpl_files/
```

`mcpltool` shows how many particles the file holds and where they were
recorded (you need the `z[cm]` column in step 6):

```bash
mcpltool -l3 data_files/mcpl_files/my_source.mcpl.gz
```

### 4. Train the model

Pick one of the two models. The steps are the same for both; the commands
below use the CFM, and the table shows what to swap for the VAE.

| | CFM | VAE |
| --- | --- | --- |
| Folder | `models/CFM` | `models/VAE` |
| Training time on a laptop CPU | a few minutes | about 10 minutes |
| Checkpoint | `data_files/models/CFM.pth` | `data_files/models/vae.pth` |
| Exported model | `CFM_sampler.onnx` | `VAE_sampler.onnx` |
| Check plot | `figures/CFM_neutron.png` | `figures/VAE_neutron.png` |

The scripts use paths relative to their own folder, so run them from the
model's folder:

```bash
cd models/CFM
python train.py --input_mcpl ../../data_files/mcpl_files/my_source.mcpl.gz
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input_mcpl` | `../../data_files/mcpl_files/ODIN.mcpl.gz` | The MCPL file to learn from. |
| `--n_particles` | `1e6` | Number of particles read from the file. |
| `--device` | `cpu` | `cpu` works everywhere; `mps` is for Apple Silicon, `cuda` for NVIDIA GPUs. |
| `--epochs` | `5` | VAE only: passes over the training data. More is slower and more accurate. |

Training prints the training and validation loss as it goes and writes the
checkpoint, plus the preprocessing table
`data_files/preprocess/gaussian_transformer.bin`. Both models write that
same table, so export a model (step 5) before training again on another MCPL
file.

### 5. Export the model

Still in the model's folder:

```bash
python eval.py --n_samples 100000 --plot
cd ../..
```

This writes the model file `Source_ML` reads,
`data_files/models/CFM_sampler.onnx`. It is self-contained: the preprocessing
is built into the model, so no other file has to travel with it
(`CFM_sampler.onnx.data` is a leftover of the export and is not needed).

`--plot` draws `--n_samples` neutrons from the model and saves their
correlation plot as `figures/CFM_neutron.png`. Look at it before moving on,
to check that the model resembles your MCPL file. The plot windows that open
have to be closed for the script to finish; set `MPLBACKEND=Agg` in front of
the command to skip them.

### 6. Copy the component to your own folder and run from there

Two files are all a new folder needs: the component and the exported model.

```bash
mkdir -p ~/my_instrument
cp mcstas_comps/Source_ML.comp ~/my_instrument/
cp data_files/models/CFM_sampler.onnx ~/my_instrument/
cd ~/my_instrument
```

`libmlbackend` is not copied. It lives in the `mcpl_torch` environment from
step 2, so that environment must be active wherever you run.

Save a minimal instrument as `~/my_instrument/My_instrument.instr`:

```
DEFINE INSTRUMENT My_instrument(string model_filename = "CFM_sampler.onnx")

TRACE

COMPONENT origin = Progress_bar()
AT (0, 0, 0) ABSOLUTE

COMPONENT source = Source_ML(ML_filename = model_filename)
AT (0, 0, 0) ABSOLUTE

COMPONENT psd = PSD_monitor(
    xwidth = 0.1, yheight = 0.1, nx = 100, ny = 100,
    filename = "psd.dat")
AT (0, 0, 60.5) ABSOLUTE

END
```

and run it:

```bash
mcrun -n 1e6 My_instrument.instr model_filename=CFM_sampler.onnx
```

The monitor sits at `z = 60.5` because the example MCPL file was recorded at
`z = 6040 cm`; put it just downstream of the `z` your own file reports in
step 3. A working run prints

```
Number of neutrons trained on: 1000000
Weight norm = 1
Detector: psd_I=... psd_ERR=... psd_N=1e+06 "psd.dat"
```

Notes:

- Give `model_filename` on the command line. `mcrun` otherwise stops to ask
  for it.
- On macOS, if `mcrun` fails with a long list of `symbol(s) not found`
  errors, put `MCSTAS_CC_OVERRIDE=/usr/bin/clang` in front of the command.
- For the VAE, copy `VAE_sampler.onnx` instead and pass
  `model_filename=VAE_sampler.onnx`.
- Instead of copying `Source_ML.comp`, you can point `mcrun` at this
  repository: `mcrun -I /path/to/mcpl_approximation/mcstas_comps ...`.

### 7. Use it in your own instrument

Copy `Source_ML.comp` and the model file next to your `.instr` file and add
the component where you would otherwise read the MCPL file:

```
COMPONENT source = Source_ML(ML_filename = "CFM_sampler.onnx")
AT (0, 0, 0) ABSOLUTE
```

- **Position.** Neutrons start at the position, direction and time stored in
  the MCPL file, relative to `Source_ML`. Place `Source_ML` at the origin of
  the instrument that wrote the MCPL file, and place everything else
  downstream of the recorded `z`.
- **Intensity.** The rays together carry the summed weight of the whole MCPL
  file, whatever `-n` is. When `--n_particles` is smaller than the file, the
  weights are scaled up to make up for the particles left out.
- **Other parameters** (`device`, `batch_size`, `verbose`) are described in
  [`mcstas_comps/README.md`](mcstas_comps/README.md).

## Benchmark

To run the whole pipeline (train, eval/export, and both McStas backends)
for both models and get a timing comparison, use:

```bash
python benchmarking/pipeline_benchmark.py
```

## Author:
Daniel Lomholt Christensen
