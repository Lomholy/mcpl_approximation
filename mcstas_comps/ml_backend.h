#pragma once

/* Unified C-linkage bridge to both ONNX Runtime and (optionally) LibTorch,
 * so a single McStas component (compiled as C) can load and sample from
 * either kind of exported model without ever seeing a backend-specific
 * type. Supersedes the separate onnx_implementation/Source_ML.comp (which
 * called the ONNX Runtime C API directly) and torchscript/torch_wrap.h
 * (which bridged LibTorch's C++-only API) with one library that does both,
 * dispatching on a resolved MLBackend.
 *
 * ONNX Runtime is a mandatory dependency of this library. LibTorch is
 * optional at build time (see CMakeLists.txt's WITH_TORCH option): a build
 * without it still provides every symbol below, but ml_load_model() returns
 * NULL (after printing an explanatory warning) for any model that resolves
 * to ML_BACKEND_TORCH.
 *
 * The inverse Gaussian-rank transform is embedded in the exported model
 * graph itself (see ../utils/data_load.py: InverseGaussRankTransform /
 * ModelWithTransform), so ml_generate_neutron()'s output is already
 * physical-space -- there is no separate transform file or step here. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLModelHandle MLModelHandle;

typedef enum {
    ML_BACKEND_AUTO = 0,  /* infer from the model path's extension */
    ML_BACKEND_ONNX = 1,
    ML_BACKEND_TORCH = 2
} MLBackend;

/* Returns one standard-normal random sample drawn using the RNG state at
 * `rng_state`. Source_ML.comp passes a thin wrapper around mcstas's own
 * randnorm() through this (mcstas's randnorm() is actually a macro,
 * `_randnorm2(_particle->randstate)` -- a per-particle/per-MPI-rank seeded
 * state threaded explicitly through every call, not a bare global
 * generator), so the noise ml_generate_neutron() feeds into the model uses
 * mcstas's own RNG stream, rather than a separately-seeded stream private
 * to this library (which, being a plain shared library rather than
 * mcstas-generated instrument code, has no access to mcstas's own RNG state
 * or its `_particle` context otherwise). `rng_state` is opaque here
 * precisely because its real type (`randstate_t*`) is defined only in
 * mcstas-generated code, not in anything this library includes; the
 * component's wrapper is what knows the real type and casts it back.
 * Preserving mcstas's actual per-particle/per-rank state (rather than
 * assuming a single fixed global stream) is also what keeps this correct
 * under upstream commit aa423dd's MPI-seeding fix. */
typedef double (*MLRandNormFn)(void* rng_state);

/* Loads the model at `path`.
 *
 * `backend` picks ONNX Runtime or LibTorch explicitly, or ML_BACKEND_AUTO
 * to infer it from `path`'s extension (.onnx -> ONNX Runtime; .pt/.pth ->
 * LibTorch). Inference failing (unrecognized extension) is a load failure.
 *
 * `device` selects where the model runs. For ONNX Runtime it is the
 * execution provider: "auto" picks CUDA, then ROCm, falling back to CPU; or
 * force one of "cuda"/"rocm"/"coreml"/"cpu". For LibTorch, "auto" picks CUDA
 * and falls back to CPU; or force one of "cuda"/"mps"/"cpu". A device that is
 * unavailable or unknown to the resolved backend falls back to CPU. May be
 * NULL, treated as "auto".
 *
 * `gpu_verbose` raises the ONNX Runtime session log severity; ignored for
 * LibTorch. `verbose` enables this library's own progress/timing prints
 * (shared by both backends).
 *
 * `batch_size` and `dim` size the noise buffer allocated internally for
 * ml_generate_neutron() (every call fills the same `batch_size * dim`
 * buffer and runs one model invocation on it).
 *
 * `randnorm_fn` is stored in the handle and called by ml_generate_neutron()
 * to fill that noise buffer -- see MLRandNormFn's comment above.
 *
 * On success, reads the model's `n_training_samples` metadata (ONNX custom
 * metadata / TorchScript extra file -- both exported under the same key by
 * utils/data_load.py) into `*n_training_samples` (0 if absent). May be
 * NULL.
 *
 * Returns NULL on failure, having already printed a diagnostic -- including
 * when `path` resolves to ML_BACKEND_TORCH but this build of the library
 * was compiled without LibTorch support, in which case the diagnostic
 * explains that and how to rebuild with it. */
MLModelHandle* ml_load_model(const char* path, MLBackend backend, const char* device,
                              int verbose, int gpu_verbose, int batch_size, int dim,
                              MLRandNormFn randnorm_fn, long* n_training_samples);

/* Generates one full batch: fills the internal `batch_size * dim` noise
 * buffer by calling `randnorm_fn(rng_state)` (this is the "initial sample"
 * that used to be drawn directly in each component's TRACE section), runs
 * it through the loaded model, and writes the `batch_size * dim` row-major
 * result into caller-allocated `output`. Dispatches internally to whichever
 * backend was resolved at load time. `rng_state` is passed straight through
 * to every `randnorm_fn` call uninterpreted -- see MLRandNormFn's comment.
 * Returns 0 on success. */
int ml_generate_neutron(MLModelHandle* handle, void* rng_state, float* output);

void ml_free_model(MLModelHandle* handle);

#ifdef __cplusplus
}
#endif
