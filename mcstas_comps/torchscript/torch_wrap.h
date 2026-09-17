#pragma once

/* C-linkage bridge to LibTorch, so a McStas component (compiled as C) can
 * load and run a TorchScript module without ever seeing a C++ type. Mirrors
 * the extern "C" pattern used by hellowrap.h/hello.h in this same directory. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TorchModelHandle TorchModelHandle;

/* Loads the TorchScript module at `path`, selecting the best available
 * device (CUDA > MPS > CPU). If the module was saved with an
 * "n_training_samples" extra file (see
 * utils/data_load.py:export_model_as_torchscript), its value is written to
 * *n_training_samples; otherwise *n_training_samples is left at 0.
 * `n_training_samples` may be NULL. Returns NULL on failure. */
TorchModelHandle* torch_load_model(const char* path, long* n_training_samples);

/* Runs the module on a (batch x dim) row-major float buffer `input` and
 * writes a (batch x dim) row-major float buffer into caller-allocated
 * `output` (must hold batch*dim floats). Handles both a module whose
 * forward() returns a single Tensor (e.g. the CFM Sampler) and one that
 * returns a tuple whose first element is the Tensor of interest (e.g. the
 * VAE). Returns 0 on success, non-zero on failure. */
int torch_run_model(TorchModelHandle* handle, const float* input, int batch, int dim, float* output);

void torch_free_model(TorchModelHandle* handle);

#ifdef __cplusplus
}
#endif
