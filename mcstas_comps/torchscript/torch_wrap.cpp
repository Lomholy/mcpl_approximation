#include "torch_wrap.h"

#include <cstdio>
#include <cstring>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

struct TorchModelHandle {
    torch::jit::script::Module module;
    torch::Device device;

    TorchModelHandle() : device(torch::kCPU) {}
};

namespace {

torch::Device pick_device()
{
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
#ifdef __APPLE__
    if (torch::hasMPS()) {
        return torch::Device(torch::kMPS);
    }
#endif
    return torch::Device(torch::kCPU);
}

} // namespace

TorchModelHandle* torch_load_model(const char* path, long* n_training_samples)
{
    auto* handle = new TorchModelHandle();

    try {
        torch::jit::ExtraFilesMap extra_files;
        extra_files["n_training_samples"] = "";

        handle->device = pick_device();
        handle->module = torch::jit::load(path, handle->device, extra_files);
        handle->module.eval();

        if (n_training_samples != nullptr) {
            const std::string& value = extra_files["n_training_samples"];
            *n_training_samples = value.empty() ? 0L : std::strtol(value.c_str(), nullptr, 10);
        }
    } catch (const c10::Error& e) {
        std::fprintf(stderr, "torch_load_model: failed to load '%s': %s\n", path, e.what());
        delete handle;
        return nullptr;
    }

    return handle;
}

int torch_run_model(TorchModelHandle* handle, const float* input, int batch, int dim, float* output)
{
    if (handle == nullptr) {
        return 1;
    }

    try {
        // from_blob does not own `input`; make a contiguous copy on the target device.
        torch::Tensor in_cpu = torch::from_blob(
            const_cast<float*>(input), {batch, dim}, torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor in = in_cpu.to(handle->device);

        std::vector<torch::jit::IValue> inputs{in};
        torch::jit::IValue result = handle->module.forward(inputs);

        torch::Tensor out_tensor;
        if (result.isTensor()) {
            out_tensor = result.toTensor();
        } else if (result.isTuple()) {
            out_tensor = result.toTuple()->elements()[0].toTensor();
        } else {
            std::fprintf(stderr, "torch_run_model: unsupported module output type\n");
            return 2;
        }

        out_tensor = out_tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
        if (out_tensor.numel() != static_cast<int64_t>(batch) * dim) {
            std::fprintf(
                stderr,
                "torch_run_model: unexpected output size %lld, expected %d\n",
                static_cast<long long>(out_tensor.numel()),
                batch * dim);
            return 3;
        }

        std::memcpy(output, out_tensor.data_ptr<float>(), static_cast<size_t>(batch) * dim * sizeof(float));
    } catch (const c10::Error& e) {
        std::fprintf(stderr, "torch_run_model: %s\n", e.what());
        return 4;
    }

    return 0;
}

void torch_free_model(TorchModelHandle* handle)
{
    delete handle;
}
