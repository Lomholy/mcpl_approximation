#include <stdio.h>
#include "../../../../../onnxruntime-osx-arm64-1.26.0/include/onnxruntime_c_api.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define CHECK(status)                                  \
    if (status != NULL) {                              \
        fprintf(stderr, "Error: %s\n",                 \
                api->GetErrorMessage(status));         \
        api->ReleaseStatus(status);                   \
        exit(1);                                      \
    }


void save_to_csv(const char* filename, float* data, int n_samples, int dim) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < dim; j++) {
            fprintf(file, "%f", data[i * dim + j]);

            if (j < dim - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main() {
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env;
    OrtStatus* status;

    // Create environment
    status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
    if (status != NULL) {
        fprintf(stderr, "Error creating environment: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return 1;
    }

    // Create session options
    OrtSessionOptions* session_options;
    status = api->CreateSessionOptions(&session_options);
    
    if (status != NULL) {
        fprintf(stderr, "Error creating session options: %s\n", api->GetErrorMessage(status));
        return 1;
    }

    // Optional: set number of threads
    CHECK(api->SetIntraOpNumThreads(session_options, 1));

    // Load model
    const char* model_path = "./velocity_field.onnx";

    OrtSession* session;
    status = api->CreateSession(env, model_path, session_options, &session);
    if (status != NULL) {
        fprintf(stderr, "Error loading model: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return 1;
    }

    printf("✅ Model loaded successfully!\n");
    // ---------------------------
    // Sampling parameters
    // ---------------------------
    int n_samples = 10000;
    int dim = 7;
    int t_steps = 256;
    float dt = 1.0f / t_steps;

    // Allocate xt
    float* xt = (float*) malloc(n_samples * dim * sizeof(float));

    // Initialize random xt ~ N(0,1)
    srand((unsigned)time(NULL));
    for (int i = 0; i < n_samples * dim; i++) {
        float u1 = ((float) rand()) / (float)RAND_MAX;
        float u2 = ((float) rand()) / (float)RAND_MAX;
        xt[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }

    // Create memory info (CPU)
    OrtMemoryInfo* memory_info;
    CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    // Input/output names (adjust if needed!)
    const char* input_names[] = {"input", "t"};
    const char* output_names[] = {"output"};

    // Shapes
    int64_t xt_shape[] = {n_samples, dim};
    int64_t t_shape[]  = {n_samples, 1};

    // Temp buffers
    float* t_vals = (float*) malloc(n_samples * sizeof(float));
    float* output   = (float*) malloc(n_samples * dim * sizeof(float));

    // ---------------------------
    // Time integration loop
    // ---------------------------
    // Create Ort tensors
    OrtValue* xt_tensor;
    OrtValue* t_tensor;
    OrtValue* output_tensor = NULL;
    printf("\nStarting time loop\n");
    for (int step = 0; step < t_steps; step++) {
      
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);


        float t_scalar = (float) step / (float) t_steps;
        printf("t=%g\n", t_scalar);

        // Fill t batch
        for (int i = 0; i < n_samples; i++) {
            t_vals[i] = t_scalar;
        }



        CHECK(api->CreateTensorWithDataAsOrtValue(
            memory_info, xt, n_samples * dim * sizeof(float),
            xt_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xt_tensor));
        CHECK(api->CreateTensorWithDataAsOrtValue(
            memory_info, t_vals, n_samples * sizeof(float),
            t_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_tensor));

        OrtValue* input_tensors[] = {xt_tensor, t_tensor};

        // Run model
        CHECK(api->Run(
            session,
            NULL,
            input_names,
            (const OrtValue* const*)input_tensors,
            2,
            output_names,
            1,
            &output_tensor
        ));

        // Get prediction data
        float* output_data;
        CHECK(api->GetTensorMutableData(output_tensor, (void**)&output_data));

        // Euler step: xt += dt * output
        for (int i = 0; i < n_samples * dim; i++) {
            xt[i] += dt * output_data[i];
        }

        
        clock_gettime(CLOCK_MONOTONIC, &end);
    
        double elapsed =
            (end.tv_sec - start.tv_sec) +
            (end.tv_nsec - start.tv_nsec) / 1e9;
    
        printf("Step %d took %.6f seconds\n", step, elapsed);
  

    }
    // Cleanup tensors
    api->ReleaseValue(xt_tensor);
    api->ReleaseValue(t_tensor);
    api->ReleaseValue(output_tensor);
    // ---------------------------
    // Print result
    // ---------------------------
    // printf("\n✅ Final samples:\n");
    // for (int i = 0; i < n_samples; i++) {
    //     printf("Sample %d: ", i);
    //     for (int j = 0; j < dim; j++) {
    //         printf("%f ", xt[i * dim + j]);
    //     }
    //     printf("\n");
    // }
    save_to_csv("C_samples.csv", xt, n_samples, dim);

    // Cleanup
    free(xt);
    free(t_vals);
    free(output);
    api->ReleaseMemoryInfo(memory_info);

    // --- Optional: Inspect model inputs ---
    // size_t num_input_nodes;
    // CHECK(api->SessionGetInputCount(session, &num_input_nodes));
    // printf("Number of inputs: %zu\n", num_input_nodes);
    //
    // for (size_t i = 0; i < num_input_nodes; i++) {
    //     char* input_name;
    //     OrtAllocator* allocator;
    //     CHECK(api->GetAllocatorWithDefaultOptions(&allocator));
    //
    //     CHECK(api->SessionGetInputName(session, i, allocator, &input_name));
    //     printf("Input %zu name: %s\n", i, input_name);
    //
    //     CHECK(api->AllocatorFree(allocator, input_name));
    // }
    //
    // size_t num_output_nodes;
    // CHECK(api->SessionGetOutputCount(session, &num_output_nodes));
    // printf("Number of outputs: %zu\n", num_output_nodes);
    //
    // for (size_t i = 0; i < num_output_nodes; i++) {
    //     char* output_name;
    //     OrtAllocator* allocator;
    //     CHECK(api->GetAllocatorWithDefaultOptions(&allocator));
    //     CHECK(api->SessionGetOutputName(session, i, allocator, &output_name))
    //     printf("Output %zu name: %s\n", i, output_name);
    //
    //     CHECK(api->AllocatorFree(allocator, output_name));
    // }

    // --- Cleanup ---
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(session_options);
    api->ReleaseEnv(env);

    return 0;
}

