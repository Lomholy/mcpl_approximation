#include "torchcall.h"

#include <chrono>
#include <iostream>
#include <vector>

torch::jit::script::Module load_model(const std::string& path)
{
    auto module = torch::jit::load(path);
    module.eval();
    module.to(torch::kMPS);
    return module;
}

torch::Tensor run_model(
    torch::jit::script::Module& module,
    const torch::Tensor& input)
{
    std::vector<torch::jit::IValue> inputs{input};

    auto output_tuple = module.forward(inputs).toTuple();

    return output_tuple->elements()[0].toTensor();
}


