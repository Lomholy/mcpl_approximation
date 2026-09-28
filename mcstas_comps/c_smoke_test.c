/* Pure C test of ml_backend.h: proves the unified ONNX/LibTorch bridge is
 * callable from plain C, the same way a McStas component's TRACE section
 * will call it -- for whichever backend the given model resolves to. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ml_backend.h"

/* Outside McStas there is no randnorm()/randstate_t to pass through as the
 * noise source, so this smoke test supplies its own (plain, unseeded-for-
 * reproducibility Box-Muller) generator -- good enough to prove the
 * load/generate/free pipeline runs, not a statistically rigorous RNG.
 * Takes (and ignores) the opaque rng_state MLRandNormFn passes through, to
 * match the real signature Source_ML.comp's wrapper implements. */
static double smoke_randnorm(void* rng_state)
{
    (void)rng_state;
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double u2 = ((double)rand()) / (double)RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s <path-to-model.onnx|.pt> [auto|onnx|torch] [device]\n", argv[0]);
        return 1;
    }

    MLBackend backend = ML_BACKEND_AUTO;
    if (argc >= 3) {
        if (strcmp(argv[2], "onnx") == 0) {
            backend = ML_BACKEND_ONNX;
        } else if (strcmp(argv[2], "torch") == 0) {
            backend = ML_BACKEND_TORCH;
        } else if (strcmp(argv[2], "auto") != 0) {
            fprintf(stderr, "unknown backend '%s' (expected auto/onnx/torch)\n", argv[2]);
            return 1;
        }
    }
    const char* device = (argc == 4) ? argv[3] : "auto";

    const int batch = 8;
    const int dim = 12;

    long n_training_samples = -1;
    MLModelHandle* handle = ml_load_model(argv[1], backend, device, /*verbose=*/1,
                                           /*gpu_verbose=*/0, batch, dim, smoke_randnorm,
                                           &n_training_samples);
    if (handle == NULL) {
        fprintf(stderr, "failed to load model '%s'\n", argv[1]);
        return 1;
    }
    printf("model loaded, n_training_samples = %ld\n", n_training_samples);

    float* output = malloc((size_t)batch * dim * sizeof(float));

    int rc = ml_generate_neutron(handle, NULL, output);
    if (rc != 0) {
        fprintf(stderr, "ml_generate_neutron failed with code %d\n", rc);
        free(output);
        ml_free_model(handle);
        return 1;
    }

    printf("output row 0:");
    for (int j = 0; j < dim; j++) {
        printf(" %f", output[j]);
    }
    printf("\n");

    free(output);
    ml_free_model(handle);

    printf("C smoke test OK\n");
    return 0;
}
