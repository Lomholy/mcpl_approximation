#include <stdio.h>
// #include "../../../../../onnxruntime-osx-arm64-1.26.0/include/onnxruntime_c_api.h"
#include "onnxruntime/core/session/onnxruntime_c_api.h"
// #include "../../../../../onnxruntime-osx-arm64-1.26.0/include/coreml_provider_factory.h"
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
    FILE* file = fopen(filename, "a");
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

int main(int argc, char* argv[]) {
  
    // ========================================================================
    // ============================ LOAD INPUT ARGS ===========================
    // ========================================================================
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n_samples>\n", argv[0]);
        return 1;
    }
    
    int n_samples = atoi(argv[1]);
    if (n_samples <= 0) {
        fprintf(stderr, "n_samples must be positive\n");
        return 1;
    }
    
    // ========================================================================
    // ======================= START ORT SESSION ==============================
    // ========================================================================

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env;

    // Create environment
    CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

    // Create session options
    OrtSessionOptions* session_options;
    CHECK(api->CreateSessionOptions(&session_options));
    // CHECK(api->SetSessionLogSeverityLevel(session_options, 0));
    // Optional: set number of threads. 0 means auto
    CHECK(api->SetIntraOpNumThreads(session_options, 0));
    CHECK(api->SetInterOpNumThreads(session_options, 0));

    // Set graph optimizations on
    CHECK(api->SetSessionGraphOptimizationLevel(
      session_options, ORT_ENABLE_ALL));

    // ========================================================================
    // =========================== LOAD MODEL =================================
    // ========================================================================
    const char* model_path = "/Users/tqv636/Phd/Projects/machine_learning/mcpl_approximation/onnx_implementation/velocity_field.onnx";
    OrtSession* session;
    CHECK(api->CreateSession(env, model_path, session_options, &session));
    printf("Model loaded successfully!\n");
    // ========================================================================
    // =========================== PREPARE SAMPLING ===========================
    // ========================================================================
    // Delete old samples
    const char *sample_file = "C_samples.csv";

    if (remove(sample_file) == 0) {
        printf("File deleted successfully.\n");
    } else {
        printf("Error: Unable to delete the file.\n");
    }
 

    int batch_size = 10000; // tune this
    if (batch_size > n_samples) batch_size = n_samples;
    int n_batches = (n_samples + batch_size - 1) / batch_size;

    
    int dim = 7;
    int t_steps = 80;
    float dt = 1.0f / t_steps;

    // Allocate xt
    float* xt = (float*) malloc(batch_size * dim * sizeof(float));
    float* save_arr = (float*) malloc(10*batch_size * dim * sizeof(float));

    // Create memory info (CPU)
    OrtMemoryInfo* memory_info;
    CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    // Input/output names (adjust if needed!)
    const char* input_names[] = {"input", "t"};
    const char* output_names[] = {"output"};

    // Shapes
    int64_t xt_shape[] = {batch_size, dim};
    int64_t t_shape[]  = {batch_size, 1};

    // Temp buffers
    float* t_vals = (float*) malloc(batch_size * sizeof(float));
    float* output   = (float*) malloc(batch_size * dim * sizeof(float));

    struct timespec tot_start, tot_end;
    clock_gettime(CLOCK_MONOTONIC, &tot_start);
    // Create Ort tensors
    OrtValue* xt_tensor;
    OrtValue* t_tensor;
    OrtValue* output_tensor;
    
    CHECK(api->CreateTensorWithDataAsOrtValue(
        memory_info, xt, batch_size * dim * sizeof(float),
        xt_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xt_tensor));

    CHECK(api->CreateTensorWithDataAsOrtValue(
        memory_info, t_vals, batch_size * sizeof(float),
        t_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_tensor));

    CHECK(api->CreateTensorWithDataAsOrtValue(
    memory_info, output, batch_size * dim * sizeof(float),
    xt_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &output_tensor));

    OrtValue* input_tensors[] = {xt_tensor, t_tensor};

    // ========================================================================
    // ======================== START SAMPLING LOOP ===========================
    // ========================================================================

    printf("\nStarting time loop for %d batches in %d steps\n", n_batches, t_steps);
    
    // buffer to accumulate all samples
    float* out_arr = (float*) malloc(batch_size*n_batches * dim * sizeof(float));
    
    for (int batch = 0; batch < n_batches; batch++) {
        int offset = batch * batch_size;
        // Reset + reinitialize xt ~ N(0,1) for this batch
        for (int i = 0; i < batch_size * dim; i++) {
            float u1 = ((float) rand()) / (float)RAND_MAX;
            float u2 = ((float) rand()) / (float)RAND_MAX;
            xt[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        }
    
        // update tensor shapes if needed
        int64_t xt_shape[] = {batch_size, dim};
        int64_t t_shape[]  = {batch_size, 1};
        for (int step = 0; step < t_steps; step++) {
    
            float t_scalar = (float) step / (float) t_steps;
    
            // fill t
            #pragma omp simd
            for (int i = 0; i < batch_size; i++) {
                t_vals[i] = t_scalar;
            }
    
            CHECK(api->Run(
                session, NULL,
                input_names,
                (const OrtValue* const*)input_tensors, 2,
                output_names, 1,
                &output_tensor
            ));
    
            float* output_data;
            CHECK(api->GetTensorMutableData(output_tensor, (void**)&output_data));
    
            #pragma omp simd
            for (int i = 0; i < batch_size * dim; i++) {
                xt[i] += dt * output_data[i];
            }
        }
    
        // store batch into final array
        memcpy(out_arr + offset * dim, xt, batch_size * dim * sizeof(float));
    
        // periodic save
        if ((batch + 1) % 10 == 0 || batch == n_batches - 1) {
            save_to_csv("C_samples.csv", out_arr, offset + batch_size, dim);
            clock_gettime(CLOCK_MONOTONIC, &tot_end);
            double elapsed =
              (tot_end.tv_sec - tot_start.tv_sec) +
              (tot_end.tv_nsec - tot_start.tv_nsec) / 1e9;
            printf("\nBatch %d! Samples generated %d, Time taken %g seconds\n",batch, batch_size*(batch+1), elapsed );
        }
    }
    
    // final cleanup
    free(out_arr);
    
    // Cleanup tensors
    api->ReleaseValue(xt_tensor);
    api->ReleaseValue(t_tensor);
    api->ReleaseValue(output_tensor);
    
    clock_gettime(CLOCK_MONOTONIC, &tot_end);
    double elapsed =
            (tot_end.tv_sec - tot_start.tv_sec) +
            (tot_end.tv_nsec - tot_start.tv_nsec) / 1e9;
    printf("\nSample generation over! Time taken %g seconds\n", elapsed );


    // Cleanup
    free(xt);
    free(t_vals);
    free(output);
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(session_options);
    api->ReleaseEnv(env);

    return 0;
}

