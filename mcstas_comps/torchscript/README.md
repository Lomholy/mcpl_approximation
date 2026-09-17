# Source_ML_torch

TorchScript counterpart of [`../onnx_implementation/Source_ML.comp`](../onnx_implementation/Source_ML.comp):
a McStas source component that samples neutrons from a Conditional Flow
Matching model. McStas compiles component code (`SHARE`/`DECLARE`/`TRACE`)
as C, but LibTorch's API is C++-only, so `torch_wrap.cpp`/`torch_wrap.h`
provide a small `extern "C"` bridge (`torch_load_model` / `torch_run_model` /
`torch_free_model`) that `Source_ML_torch.comp` calls instead of talking to
LibTorch directly.

## Files

- `Source_ML_torch.comp` — the McStas component.
- `torch_wrap.h` / `torch_wrap.cpp` — the C-linkage LibTorch bridge.
- `CMakeLists.txt` — builds `torch_wrap.cpp` into `libtorchwrap` and installs
  it (with its header) into the active environment, and builds
  `c_smoke_test.c`, a pure-C sanity check of the bridge.
- `c_smoke_test.c` — loads and runs a TorchScript module through the C API
  only, independent of McStas.
- `torch_test.py` — exports a `VelocityField`/`Sampler` (see
  `../../models/CFM/model.py`) to TorchScript via
  `utils/data_load.py:export_model_as_torchscript`, producing
  `CFM_sampler.pt`. Uses a real checkpoint at
  `../../data_files/models/CFM.pth` if present, otherwise exports an
  untrained model for pipeline smoke-testing.
- `make_smoke_transformer.py` — writes a synthetic
  `gaussian_transformer_smoke.bin`, standing in for the real
  `../../data_files/preprocess/gaussian_transformer.bin` when that isn't
  available. Values are unconstrained random Gaussians, so anything sampled
  through it is not physically meaningful — smoke-testing only.
- `Test_Source_ML_torch.instr` — minimal instrument wiring the component to
  a `PSD_monitor`, for smoke-testing.

`CFM_sampler.pt` and `gaussian_transformer_smoke.bin` are generated
artifacts and intentionally not committed, matching how
`../onnx_implementation` doesn't check in `.onnx`/`.bin` files either.

## One-time environment setup

```bash
micromamba create -n mcpl_torch -c conda-forge \
    python=3.11 pytorch-cpu cmake make cxx-compiler c-compiler \
    onnx onnxruntime numpy mcstas
micromamba activate mcpl_torch
pip install np2mcpl
pip install "numpy<2"   # np2mcpl's wheel is built against the NumPy 1.x ABI
```

## Build and install the wrapper

`Source_ML_torch.comp`'s `DEPENDENCY` line is just `-ltorchwrap`; it relies
on mcstas's own conda-aware `CFLAGS` (from `mccode_config.json`) for
`-I`/`-L`/`-rpath` into the active environment, so `libtorchwrap` and
`torch_wrap.h` need to be installed there:

```bash
cd mcstas_comps/torchscript
TORCH_CMAKE=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" ..
make -j4
cmake --install .
cd ..
```

If your system's compiler toolchain is newer than the one conda-forge's
`cxx-compiler`/`c-compiler` packages bundle (macOS with a recent Xcode
Command Line Tools is the case we hit — conda's linker can't parse the SDK's
`.tbd` files and the build fails with `ld: warning: ... malformed file`),
configure with the system compiler instead:

```bash
CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" ..
```

Optionally sanity-check the bridge on its own, independent of McStas:

```bash
DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:build" ./build/c_smoke_test ./CFM_sampler.pt
```

(needs `CFM_sampler.pt` — see next section.)

## Generate the smoke-test model and transformer

```bash
python torch_test.py                # writes CFM_sampler.pt
python make_smoke_transformer.py     # writes gaussian_transformer_smoke.bin
```

## Run the instrument

```bash
MCSTAS_CC_OVERRIDE=/usr/bin/clang mcrun -n 1000 Test_Source_ML_torch.instr \
    model_filename=CFM_sampler.pt transformer_filename=gaussian_transformer_smoke.bin
```

`MCSTAS_CC_OVERRIDE` is only needed if `mcrun`'s own compile step hits the
same conda-linker-vs-newer-SDK issue as above — if you hit a cascade of
`"symbol(s) not found"` errors for basic C functions (`malloc`, `printf`,
`rand`, ...), that's this issue, not a problem with the component itself.

With the untrained smoke-test model and synthetic transformer this only
confirms the load/run/inverse-transform/emit pipeline executes; it does not
validate physics. For that, point `model_filename` at a TorchScript export
of a real checkpoint (produced by `torch_test.py` once
`../../data_files/models/CFM.pth` exists) and `transformer_filename` at the
real `../../data_files/preprocess/gaussian_transformer.bin`.
