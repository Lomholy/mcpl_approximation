#include <torch/script.h>
#include <torch/torch.h>

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

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;

    try {
        module = load_model(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "Model loaded successfully\n";

    auto device = torch::Device(torch::kMPS);
    for (int i = 0; i<100; i++){
        torch::Tensor input =
            torch::randn({10000, 12}, torch::TensorOptions().device(device));

        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor output = run_model(module, input);
        output.cpu();


        // Ensure execution is complete before stopping timer
        auto end = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end - start)
                      .count();

        std::cout << "Inference time: " << ms << " ms\n";
    }

    return 0;
}
