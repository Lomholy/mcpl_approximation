# Source_ML

A single McStas source component that samples neutrons from a machine
learning model exported to either ONNX or TorchScript, dispatching between
the two backends at runtime through one C++ library, `libmlbackend`
(`ml_backend.h`/`ml_backend.cpp` in this directory).

This replaces the former separate `mcstas_comps/onnx_implementation`
(ONNX-only `Source_ML`) and `mcstas_comps/torchscript` (TorchScript-only
`Source_ML_torch`) components. Those two `.comp` files were otherwise
near-identical duplicates -- the only real difference was which C API they
called into -- so this component keeps one copy of that logic and lets
`ML_filename`'s extension (or an explicit `backend` parameter) pick the
backend.

**No transform file.** The inverse Gaussian-rank transform is embedded in
the exported model graph itself (`utils/data_load.py:ModelWithTransform`),
so `Source_ML`'s output is already physical-space. There is nothing like a
`transformer_filename` parameter here.

## Files

- `Source_ML.comp` -- the McStas component. Takes `ML_filename`, `backend`
  (`"auto"`/`"onnx"`/`"torch"`), `device` (ONNX Runtime execution provider,
  ignored for `backend="torch"`), `batch_size`, `verbose`, `GPU_verbose`.
- `ml_backend.h` / `ml_backend.cpp` -- the unified C-linkage bridge. ONNX
  Runtime is a mandatory dependency; LibTorch is optional (see below).
- `CMakeLists.txt` -- builds `ml_backend.cpp` into `libmlbackend` and
  installs it (with its header) into the active environment; auto-detects
  LibTorch via `WITH_TORCH` (`AUTO`/`ON`/`OFF`). Also builds `c_smoke_test`,
  a pure-C sanity check of the library, independent of McStas.
- `c_smoke_test.c` -- loads and runs a model (either backend) through the C
  API only.
- `export_onnx_smoke.py` / `export_torch_smoke.py` -- export a
  `VelocityField`/`Sampler` (see `../../models/CFM/model.py`) to
  `CFM_sampler.onnx` / `CFM_sampler.pt` via `utils/data_load.py`. Use a real
  checkpoint at `../../data_files/models/CFM.pth` if present, otherwise
  export an untrained model for pipeline smoke-testing.
- `make_smoke_transformer.py` -- writes a synthetic
  `gaussian_transformer_smoke.bin`, standing in for the real
  `../../data_files/preprocess/gaussian_transformer.bin` when that isn't
  available. This is consumed by the two export scripts above (which bake it
  into the exported model), not by `Source_ML.comp` itself -- see "No
  transform file" above. Values are unconstrained random Gaussians, so
  anything sampled through a smoke-exported model is not physically
  meaningful.
- `Test_Source_ML.instr` -- minimal instrument wiring the component to a
  `PSD_monitor`, for smoke-testing either backend.

`CFM_sampler.onnx`, `CFM_sampler.pt`, and `gaussian_transformer_smoke.bin`
are generated artifacts and intentionally not committed.

## One-time environment setup

```bash
mamba env create -f ../../environment.yml
mamba activate mcpl_torch
```

`environment.yml` installs `onnxruntime-cpp` (the C API headers/lib that
`libmlbackend` needs -- plain `onnxruntime` from conda-forge only ships the
Python bindings) and `pytorch-cpu` (which ships LibTorch's headers/libs and
`torch.utils.cmake_prefix_path`). **LibTorch is optional for `libmlbackend`
itself** -- if you only care about the ONNX backend, you can build in an
environment without `pytorch-cpu` at all; `libmlbackend` will build fine and
`Source_ML` will work for `.onnx` models. Pointing it at a `.pt`/`.pth` file
in that case fails at `INITIALIZE` with a message telling you to install
PyTorch and rebuild, not a crash.

## Build and install the library

`Source_ML.comp`'s `DEPENDENCY` line is just `-lmlbackend`; it relies on
mcstas's own conda-aware `CFLAGS` (from `mccode_config.json`) for
`-I`/`-L`/`-rpath` into the active environment, so `libmlbackend` and
`ml_backend.h` need to be installed there:

```bash
cd mcstas_comps/Source_ML
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" ..
make -j4
cmake --install .
cd ..
```

This auto-detects LibTorch (`WITH_TORCH=AUTO`, the default): if
`pytorch-cpu` is installed and `torch.utils.cmake_prefix_path` is on
`CMAKE_PREFIX_PATH`, the build includes TorchScript support; otherwise it
silently builds ONNX-only (check the `cmake` configure output for which one
happened). To require or forbid TorchScript support explicitly:

```bash
# Force TorchScript support on, failing configure if LibTorch isn't found:
TORCH_CMAKE=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DWITH_TORCH=ON ..

# Force an ONNX-only build even if LibTorch is installed:
cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DWITH_TORCH=OFF ..
```

If your system's compiler toolchain is newer than the one conda-forge's
`cxx-compiler`/`c-compiler` packages bundle (macOS with a recent Xcode
Command Line Tools is the case we hit -- conda's linker can't parse the
SDK's `.tbd` files and the build fails with `ld: warning: ... malformed
file`), configure with the system compiler instead:

```bash
CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" ..
```

Optionally sanity-check the library on its own, independent of McStas:

```bash
DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:build" ./build/c_smoke_test ./CFM_sampler.onnx onnx
DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:build" ./build/c_smoke_test ./CFM_sampler.pt torch
```

(needs the smoke-test model files -- see next section.)

## Generate the smoke-test models

```bash
python export_onnx_smoke.py    # writes CFM_sampler.onnx
python export_torch_smoke.py   # writes CFM_sampler.pt (only useful with a Torch-enabled libmlbackend)
```

Both scripts call `make_smoke_transformer.write_smoke_transformer()`
themselves if `../../data_files/preprocess/gaussian_transformer.bin` isn't
present, so you don't need to run it separately.

## Run the instrument

```bash
MCSTAS_CC_OVERRIDE=/usr/bin/clang mcrun -n 1000 Test_Source_ML.instr \
    model_filename=CFM_sampler.onnx backend=auto
MCSTAS_CC_OVERRIDE=/usr/bin/clang mcrun -n 1000 Test_Source_ML.instr \
    model_filename=CFM_sampler.pt backend=auto
```

`MCSTAS_CC_OVERRIDE` is only needed if `mcrun`'s own compile step hits the
same conda-linker-vs-newer-SDK issue as above -- if you hit a cascade of
`"symbol(s) not found"` errors for basic C functions (`malloc`, `printf`,
`rand`, ...), that's this issue, not a problem with the component itself.

With the untrained smoke-test model this only confirms the
load/generate/emit pipeline executes; it does not validate physics. For
that, point `model_filename` at a real trained export (produced by
`export_onnx_smoke.py`/`export_torch_smoke.py` once
`../../data_files/models/CFM.pth` and
`../../data_files/preprocess/gaussian_transformer.bin` exist from the normal
training/eval pipeline -- see the repo root README).
