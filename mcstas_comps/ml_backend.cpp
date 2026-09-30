#include "ml_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#include "onnxruntime/core/session/onnxruntime_c_api.h"

#include <torch/script.h>
#include <torch/torch.h>

namespace {

double ml_walltime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

MLBackend infer_backend_from_extension(const char* path)
{
    const char* dot = strrchr(path, '.');
    if (dot == nullptr) return ML_BACKEND_AUTO;
    if (strcmp(dot, ".onnx") == 0) return ML_BACKEND_ONNX;
    if (strcmp(dot, ".pt") == 0 || strcmp(dot, ".pth") == 0) return ML_BACKEND_TORCH;
    return ML_BACKEND_AUTO;
}

// ============================================================================
// ONNX Runtime backend (mandatory)
// ============================================================================

struct OnnxState {
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* session_options = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memory_info = nullptr;
    OrtValue* input_tensor = nullptr;
    OrtValue* output_tensor = nullptr;
    const char* input_names[1] = {"input"};
    const char* output_names[1] = {"output"};
    float* output_data = nullptr;  // backing buffer of output_tensor
};

// --------------------------------------------------------------------------
// Hardware auto-detection: try to append a GPU-accelerated execution
// provider, preferring CUDA, then ROCm, then CoreML, falling back to CPU
// (ORT's own default) if none of those are compiled in or available on this
// machine. Ported verbatim from the former onnx_implementation/Source_ML.comp.
// --------------------------------------------------------------------------
int ep_provider_available(char** avail, int n_avail, const char* name)
{
    for (int i = 0; i < n_avail; i++) {
        if (strcmp(avail[i], name) == 0) return 1;
    }
    return 0;
}

OrtStatus* ep_append_cuda(const OrtApi* api, OrtSessionOptions* session_options)
{
    OrtCUDAProviderOptions cuda_options;
    memset(&cuda_options, 0, sizeof(cuda_options));
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.gpu_mem_limit = (size_t)-1;
    cuda_options.do_copy_in_default_stream = 1;
    return api->SessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_options);
}

OrtStatus* ep_append_rocm(const OrtApi* api, OrtSessionOptions* session_options)
{
    OrtROCMProviderOptions rocm_options;
    memset(&rocm_options, 0, sizeof(rocm_options));
    rocm_options.device_id = 0;
    rocm_options.gpu_mem_limit = (size_t)-1;
    rocm_options.do_copy_in_default_stream = 1;
    return api->SessionOptionsAppendExecutionProvider_ROCM(session_options, &rocm_options);
}

OrtStatus* ep_append_coreml(const OrtApi* api, OrtSessionOptions* session_options)
{
    const char* keys[] = {"ModelFormat"};
    const char* values[] = {"MLProgram"};
    return api->SessionOptionsAppendExecutionProvider(session_options, "CoreML", keys, values, 1);
}

// Attempts to append the named provider ("cuda"/"rocm"/"coreml"/"cpu").
// Returns 1 if the session will use it (for "cpu" this is trivially true --
// there is no EP to append, ORT always has a CPU fallback), 0 if the
// provider is unavailable or failed to append.
int ep_try_append(const OrtApi* api, OrtSessionOptions* session_options,
                   char** avail, int n_avail, const char* name, int verbose)
{
    OrtStatus* status = NULL;

    if (strcmp(name, "cpu") == 0) {
        if (verbose) printf("ml_backend (onnx): using cpu execution provider\n");
        return 1;
    } else if (strcmp(name, "cuda") == 0) {
        if (!ep_provider_available(avail, n_avail, "CUDAExecutionProvider")) return 0;
        status = ep_append_cuda(api, session_options);
    } else if (strcmp(name, "rocm") == 0) {
        if (!ep_provider_available(avail, n_avail, "ROCMExecutionProvider")) return 0;
        status = ep_append_rocm(api, session_options);
    } else if (strcmp(name, "coreml") == 0) {
        if (!ep_provider_available(avail, n_avail, "CoreMLExecutionProvider")) return 0;
        status = ep_append_coreml(api, session_options);
    } else {
        fprintf(stderr, "ml_backend (onnx): unknown device '%s'\n", name);
        return 0;
    }

    if (status != NULL) {
        if (verbose) {
            fprintf(stderr, "ml_backend (onnx): failed to append %s execution provider: %s\n",
                    name, api->GetErrorMessage(status));
        }
        api->ReleaseStatus(status);
        return 0;
    }
    if (verbose) printf("ml_backend (onnx): using %s execution provider\n", name);
    return 1;
}

#define ORT_CHECK(expr)                                                       \
    do {                                                                      \
        OrtStatus* _status = (expr);                                          \
        if (_status != NULL) {                                                \
            fprintf(stderr, "ml_backend (onnx): %s\n", st.api->GetErrorMessage(_status)); \
            st.api->ReleaseStatus(_status);                                   \
            return false;                                                     \
        }                                                                     \
    } while (0)

bool onnx_load(OnnxState& st, const char* path, const char* device, int verbose,
               int gpu_verbose, int batch_size, int dim, float* noise,
               long* n_training_samples)
{
    st.api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    ORT_CHECK(st.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mlbackend", &st.env));
    ORT_CHECK(st.api->CreateSessionOptions(&st.session_options));

    // ------------------------------------------------------------------
    // Hardware auto-detection
    // ------------------------------------------------------------------
    char** avail_providers = NULL;
    int n_avail_providers = 0;
    ORT_CHECK(st.api->GetAvailableProviders(&avail_providers, &n_avail_providers));
    if (verbose) {
        printf("ml_backend (onnx): available execution providers:");
        for (int i = 0; i < n_avail_providers; i++) printf(" %s", avail_providers[i]);
        printf("\n");
    }

    // CoreML is deliberately excluded from "auto": running this library's
    // models (which always have the inverse Gaussian-rank transform's
    // Floor/GatherElements ops embedded under a dynamic batch shape --
    // see InverseGaussRankTransform in utils/data_load.py) through CoreML
    // was verified (2026-09-28, onnxruntime 1.30.0, macOS 11 target on
    // Apple Silicon) to crash inside Apple's own CoreML/BNNS runtime
    // (EXC_BAD_ACCESS/SIGBUS, misaligned access deep in
    // Espresso::BNNSEngine, called from MLNeuralNetworkEngine) during
    // inference, not merely emit the harmless "unbounded dimension"
    // load-time diagnostic this component used to assume was the whole
    // story. It remains selectable via device="coreml" for anyone who has
    // verified their model/hardware/OS combination doesn't hit this, but
    // "auto" must not pick something that can crash the whole process.
    const char* device_name = (device != NULL && device[0] != '\0') ? device : "auto";
    if (strcmp(device_name, "auto") == 0) {
        static const char* device_priority[] = {"cuda", "rocm", "cpu"};
        int n_device_priority = sizeof(device_priority) / sizeof(device_priority[0]);
        for (int i = 0; i < n_device_priority; i++) {
            if (ep_try_append(st.api, st.session_options, avail_providers, n_avail_providers,
                               device_priority[i], verbose)) {
                break;
            }
        }
    } else if (!ep_try_append(st.api, st.session_options, avail_providers, n_avail_providers,
                               device_name, verbose)) {
        fprintf(stderr, "ml_backend (onnx): requested device '%s' unavailable; falling back to CPU\n", device_name);
    }
    OrtStatus* release_status = st.api->ReleaseAvailableProviders(avail_providers, n_avail_providers);
    if (release_status != NULL) st.api->ReleaseStatus(release_status);

    if (gpu_verbose) {
        ORT_CHECK(st.api->SetSessionLogSeverityLevel(st.session_options, 0));
    }
    ORT_CHECK(st.api->SetIntraOpNumThreads(st.session_options, 0));
    ORT_CHECK(st.api->SetInterOpNumThreads(st.session_options, 0));
    ORT_CHECK(st.api->SetSessionGraphOptimizationLevel(st.session_options, ORT_ENABLE_ALL));

    // ------------------------------------------------------------------
    // Load model
    // ------------------------------------------------------------------
    ORT_CHECK(st.api->CreateSession(st.env, path, st.session_options, &st.session));
    if (verbose) printf("ml_backend (onnx): model loaded successfully!\n");

    if (n_training_samples != NULL) *n_training_samples = 0;
    OrtModelMetadata* metadata = NULL;
    ORT_CHECK(st.api->SessionGetModelMetadata(st.session, &metadata));
    OrtAllocator* allocator = NULL;
    ORT_CHECK(st.api->GetAllocatorWithDefaultOptions(&allocator));
    char* value = NULL;
    ORT_CHECK(st.api->ModelMetadataLookupCustomMetadataMap(metadata, allocator, "n_training_samples", &value));
    if (value) {
        if (n_training_samples != NULL) *n_training_samples = atol(value);
        allocator->Free(allocator, value);
    }
    st.api->ReleaseModelMetadata(metadata);

    // ------------------------------------------------------------------
    // Prepare sampling buffers
    // ------------------------------------------------------------------
    ORT_CHECK(st.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &st.memory_info));

    int64_t shape[] = {batch_size, dim};
    st.output_data = (float*)malloc((size_t)batch_size * dim * sizeof(float));

    ORT_CHECK(st.api->CreateTensorWithDataAsOrtValue(
        st.memory_info, noise, (size_t)batch_size * dim * sizeof(float),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &st.input_tensor));
    ORT_CHECK(st.api->CreateTensorWithDataAsOrtValue(
        st.memory_info, st.output_data, (size_t)batch_size * dim * sizeof(float),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &st.output_tensor));

    return true;
}

#undef ORT_CHECK

int onnx_run(OnnxState& st, float* output, int batch_size, int dim)
{
    OrtStatus* status = st.api->Run(
        st.session, NULL, st.input_names,
        (const OrtValue* const*)&st.input_tensor, 1,
        st.output_names, 1, &st.output_tensor);
    if (status != NULL) {
        fprintf(stderr, "ml_backend (onnx): Run failed: %s\n", st.api->GetErrorMessage(status));
        st.api->ReleaseStatus(status);
        return 1;
    }
    memcpy(output, st.output_data, (size_t)batch_size * dim * sizeof(float));
    return 0;
}

void onnx_free(OnnxState& st)
{
    if (st.api == NULL) return;
    if (st.input_tensor) st.api->ReleaseValue(st.input_tensor);
    if (st.output_tensor) st.api->ReleaseValue(st.output_tensor);
    free(st.output_data);
    if (st.memory_info) st.api->ReleaseMemoryInfo(st.memory_info);
    if (st.session) st.api->ReleaseSession(st.session);
    if (st.session_options) st.api->ReleaseSessionOptions(st.session_options);
    if (st.env) st.api->ReleaseEnv(st.env);
}

// ============================================================================
// LibTorch backend (mandatory)
// ============================================================================

struct TorchState {
    torch::jit::script::Module module;
    torch::Device device;

    TorchState() : device(torch::kCPU) {}
};

// "auto" picks CUDA when available and CPU otherwise. MPS is only used when
// asked for by name.
torch::Device pick_torch_device(const char* device, int verbose)
{
    const char* name = (device != NULL && device[0] != '\0') ? device : "auto";

    if (strcmp(name, "auto") == 0 || strcmp(name, "cuda") == 0) {
        if (torch::cuda::is_available()) {
            if (verbose) printf("ml_backend (torch): using cuda\n");
            return torch::Device(torch::kCUDA);
        }
        if (strcmp(name, "cuda") == 0) {
            fprintf(stderr, "ml_backend (torch): requested device 'cuda' unavailable; falling back to CPU\n");
        }
    } else if (strcmp(name, "mps") == 0) {
#ifdef __APPLE__
        if (torch::hasMPS()) {
            if (verbose) printf("ml_backend (torch): using mps\n");
            return torch::Device(torch::kMPS);
        }
#endif
        fprintf(stderr, "ml_backend (torch): requested device 'mps' unavailable; falling back to CPU\n");
    } else if (strcmp(name, "cpu") != 0) {
        fprintf(stderr, "ml_backend (torch): unknown device '%s'; falling back to CPU\n", name);
    }

    if (verbose) printf("ml_backend (torch): using cpu\n");
    return torch::Device(torch::kCPU);
}

bool torch_load(TorchState& st, const char* path, const char* device, int verbose,
                long* n_training_samples)
{
    try {
        torch::jit::ExtraFilesMap extra_files;
        extra_files["n_training_samples"] = "";

        st.device = pick_torch_device(device, verbose);
        st.module = torch::jit::load(path, st.device, extra_files);
        st.module.eval();

        if (n_training_samples != nullptr) {
            const std::string& value = extra_files["n_training_samples"];
            *n_training_samples = value.empty() ? 0L : std::strtol(value.c_str(), nullptr, 10);
        }
    } catch (const c10::Error& e) {
        fprintf(stderr, "ml_backend (torch): failed to load '%s': %s\n", path, e.what());
        return false;
    }
    return true;
}

int torch_run(TorchState& st, const float* input, int batch, int dim, float* output)
{
    try {
        // from_blob does not own `input`; make a contiguous copy on the target device.
        torch::Tensor in_cpu = torch::from_blob(
            const_cast<float*>(input), {batch, dim}, torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor in = in_cpu.to(st.device);

        std::vector<torch::jit::IValue> inputs{in};
        torch::jit::IValue result = st.module.forward(inputs);

        torch::Tensor out_tensor;
        if (result.isTensor()) {
            out_tensor = result.toTensor();
        } else if (result.isTuple()) {
            out_tensor = result.toTuple()->elements()[0].toTensor();
        } else {
            fprintf(stderr, "ml_backend (torch): unsupported module output type\n");
            return 2;
        }

        out_tensor = out_tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
        if (out_tensor.numel() != static_cast<int64_t>(batch) * dim) {
            fprintf(stderr, "ml_backend (torch): unexpected output size %lld, expected %d\n",
                    static_cast<long long>(out_tensor.numel()), batch * dim);
            return 3;
        }

        memcpy(output, out_tensor.data_ptr<float>(), static_cast<size_t>(batch) * dim * sizeof(float));
    } catch (const c10::Error& e) {
        fprintf(stderr, "ml_backend (torch): %s\n", e.what());
        return 4;
    }
    return 0;
}

}  // namespace

// ============================================================================
// Unified handle and public API
// ============================================================================

struct MLModelHandle {
    MLBackend backend;
    int batch_size;
    int dim;
    int verbose;
    MLRandNormFn randnorm;
    float* noise;

    OnnxState onnx;
    TorchState torch_state;
};

extern "C" {

MLModelHandle* ml_load_model(const char* path, MLBackend backend, const char* device,
                              int verbose, int gpu_verbose, int batch_size, int dim,
                              MLRandNormFn randnorm_fn, long* n_training_samples)
{
    if (n_training_samples != NULL) *n_training_samples = 0;

    MLBackend resolved = backend;
    if (resolved == ML_BACKEND_AUTO) {
        resolved = infer_backend_from_extension(path);
        if (resolved == ML_BACKEND_AUTO) {
            fprintf(stderr,
                    "ml_load_model: cannot infer backend from '%s' (expected a .onnx, "
                    ".pt, or .pth extension); pass backend=\"onnx\" or backend=\"torch\" explicitly\n",
                    path);
            return nullptr;
        }
    }

    auto* handle = new MLModelHandle();
    handle->backend = resolved;
    handle->batch_size = batch_size;
    handle->dim = dim;
    handle->verbose = verbose;
    handle->randnorm = randnorm_fn;
    handle->noise = (float*)malloc((size_t)batch_size * dim * sizeof(float));

    bool ok = false;
    if (resolved == ML_BACKEND_ONNX) {
        ok = onnx_load(handle->onnx, path, device, verbose, gpu_verbose, batch_size, dim,
                        handle->noise, n_training_samples);
    } else {
        ok = torch_load(handle->torch_state, path, device, verbose, n_training_samples);
    }

    if (!ok) {
        free(handle->noise);
        delete handle;
        return nullptr;
    }

    return handle;
}

int ml_generate_neutron(MLModelHandle* handle, void* rng_state, float* output)
{
    if (handle == nullptr) return 1;

    double t0 = handle->verbose ? ml_walltime() : 0.0;
    for (int i = 0; i < handle->batch_size * handle->dim; i++) {
        handle->noise[i] = (float)handle->randnorm(rng_state);
    }
    if (handle->verbose) {
        printf("Gaussian init: %.3f ms\n", 1000.0 * (ml_walltime() - t0));
    }

    double r0 = handle->verbose ? ml_walltime() : 0.0;
    int rc;
    if (handle->backend == ML_BACKEND_ONNX) {
        rc = onnx_run(handle->onnx, output, handle->batch_size, handle->dim);
    } else {
        rc = torch_run(handle->torch_state, handle->noise, handle->batch_size, handle->dim, output);
    }
    if (handle->verbose && rc == 0) {
        printf("Model Run: %.3f s\n", ml_walltime() - r0);
    }
    return rc;
}

void ml_free_model(MLModelHandle* handle)
{
    if (handle == nullptr) return;
    if (handle->backend == ML_BACKEND_ONNX) {
        onnx_free(handle->onnx);
    }
    // The LibTorch backend's torch::jit::script::Module/torch::Device clean
    // themselves up via TorchState's destructor when `handle` is deleted.
    free(handle->noise);
    delete handle;
}

}  // extern "C"
