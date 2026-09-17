/* Pure C test of torch_wrap.h: proves the LibTorch bridge is callable from
 * plain C, the same way a McStas component's TRACE section will call it. */
#include <stdio.h>
#include <stdlib.h>

#include "torch_wrap.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <path-to-scripted-model.pt>\n", argv[0]);
        return 1;
    }

    long n_training_samples = -1;
    TorchModelHandle* handle = torch_load_model(argv[1], &n_training_samples);
    if (handle == NULL) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    printf("model loaded, n_training_samples = %ld\n", n_training_samples);

    const int batch = 8;
    const int dim = 12;
    float* input = malloc(batch * dim * sizeof(float));
    float* output = malloc(batch * dim * sizeof(float));

    for (int i = 0; i < batch * dim; i++) {
        input[i] = (float)rand() / (float)RAND_MAX;
    }

    int rc = torch_run_model(handle, input, batch, dim, output);
    if (rc != 0) {
        fprintf(stderr, "torch_run_model failed with code %d\n", rc);
        return 1;
    }

    printf("output row 0:");
    for (int j = 0; j < dim; j++) {
        printf(" %f", output[j]);
    }
    printf("\n");

    free(input);
    free(output);
    torch_free_model(handle);

    printf("C smoke test OK\n");
    return 0;
}
