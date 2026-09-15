#pragma once

#include <string>

#include <torch/script.h>
#include <torch/torch.h>

torch::jit::script::Module load_model(const std::string& path);

torch::Tensor run_model(
    torch::jit::script::Module& module,
    const torch::Tensor& input);
