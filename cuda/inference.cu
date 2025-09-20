#include "neuralNetwork.cuh"

// Inference-specific structure (lighter than training structure)
typedef struct {
    // Weights for 2 hidden layers
    float *d_weights1;  // INPUT_SIZE → HIDDEN_SIZE1 (784 → 1568)
    float *d_weights2;  // HIDDEN_SIZE1 → HIDDEN_SIZE2 (1568 → 784) 
    float *d_weights3;  // HIDDEN_SIZE2 → OUTPUT_SIZE (784 → 10)
    
    // Biases for 2 hidden layers
    float *d_bias1;     // HIDDEN_SIZE1 (1568)
    float *d_bias2;     // HIDDEN_SIZE2 (784)
    float *d_bias3;     // OUTPUT_SIZE (10)
    
    // Layer outputs (activations) - only need single sample
    float *d_fc1_output;     // 1 × HIDDEN_SIZE1
    float *d_fc2_output;     // 1 × HIDDEN_SIZE2
    float *d_fc3_output;     // 1 × OUTPUT_SIZE
    
    // Input buffer
    float *d_input;
    float *h_output;
    
    cublasHandle_t cublas_handle;
} InferenceNetwork;

void initialize_inference_network(InferenceNetwork *nn) {
    // Allocate weights and biases
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias3, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate layer outputs (single sample)
    CUDA_CHECK(cudaMalloc(&nn->d_fc1_output, HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc2_output, HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc3_output, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate input and output buffers
    CUDA_CHECK(cudaMalloc(&nn->d_input, INPUT_SIZE * sizeof(float)));
    nn->h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
}

void free_inference_network(InferenceNetwork *nn) {
    cudaFree(nn->d_weights1);
    cudaFree(nn->d_weights2);
    cudaFree(nn->d_weights3);
    cudaFree(nn->d_bias1);
    cudaFree(nn->d_bias2);
    cudaFree(nn->d_bias3);
    cudaFree(nn->d_fc1_output);
    cudaFree(nn->d_fc2_output);
    cudaFree(nn->d_fc3_output);
    cudaFree(nn->d_input);
    free(nn->h_output);
    cublasDestroy(nn->cublas_handle);
}

void load_weights_for_inference(InferenceNetwork *nn, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for reading\n", filename);
        exit(EXIT_FAILURE);
    }
    
    // Read and verify network architecture
    int file_input_size, file_hidden1, file_hidden2, file_output_size;
    fread(&file_input_size, sizeof(int), 1, f);
    fread(&file_hidden1, sizeof(int), 1, f);
    fread(&file_hidden2, sizeof(int), 1, f);
    fread(&file_output_size, sizeof(int), 1, f);
    
    if (file_input_size != INPUT_SIZE || file_hidden1 != HIDDEN_SIZE1 || 
        file_hidden2 != HIDDEN_SIZE2 || file_output_size != OUTPUT_SIZE) {
        fprintf(stderr, "Error: Network architecture mismatch in %s\n", filename);
        fprintf(stderr, "Expected: %d->%d->%d->%d, Got: %d->%d->%d->%d\n", 
                INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE,
                file_input_size, file_hidden1, file_hidden2, file_output_size);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    
    // Create temporary host buffers
    float *h_weights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float));
    float *h_weights2 = (float*)malloc(HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float));
    float *h_weights3 = (float*)malloc(HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float));
    float *h_bias1 = (float*)malloc(HIDDEN_SIZE1 * sizeof(float));
    float *h_bias2 = (float*)malloc(HIDDEN_SIZE2 * sizeof(float));
    float *h_bias3 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Read weights and biases from file
    fread(h_weights1, sizeof(float), INPUT_SIZE * HIDDEN_SIZE1, f);
    fread(h_weights2, sizeof(float), HIDDEN_SIZE1 * HIDDEN_SIZE2, f);
    fread(h_weights3, sizeof(float), HIDDEN_SIZE2 * OUTPUT_SIZE, f);
    fread(h_bias1, sizeof(float), HIDDEN_SIZE1, f);
    fread(h_bias2, sizeof(float), HIDDEN_SIZE2, f);
    fread(h_bias3, sizeof(float), OUTPUT_SIZE, f);
    
    fclose(f);
    
    // Copy weights from host to device
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->d_weights3, h_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1, HIDDEN_SIZE1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2, HIDDEN_SIZE2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->d_bias3, h_bias3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary buffers
    free(h_weights1);
    free(h_weights2);
    free(h_weights3);
    free(h_bias1);
    free(h_bias2);
    free(h_bias3);
    
    printf("Model weights loaded from %s\n", filename);
}

__global__ void bias_add_kernel_single(float *x, float *bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += bias[idx];
    }
}

__global__ void relu_kernel_single(float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void softmax_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Find max for numerical stability
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = expf(x[i] - max_val);
            sum += x[i];
        }
        
        // Normalize
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }
}

int predict(InferenceNetwork *nn, float *input_image) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(nn->d_input, input_image, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Layer 1: INPUT_SIZE → HIDDEN_SIZE1
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(nn->cublas_handle, CUBLAS_OP_N, HIDDEN_SIZE1, INPUT_SIZE,
                             &alpha, nn->d_weights1, HIDDEN_SIZE1, nn->d_input, 1, &beta, nn->d_fc1_output, 1));
    
    // Add bias and ReLU
    bias_add_kernel_single<<<(HIDDEN_SIZE1 + 255) / 256, 256>>>(nn->d_fc1_output, nn->d_bias1, HIDDEN_SIZE1);
    relu_kernel_single<<<(HIDDEN_SIZE1 + 255) / 256, 256>>>(nn->d_fc1_output, HIDDEN_SIZE1);
    
    // Layer 2: HIDDEN_SIZE1 → HIDDEN_SIZE2
    CUBLAS_CHECK(cublasSgemv(nn->cublas_handle, CUBLAS_OP_N, HIDDEN_SIZE2, HIDDEN_SIZE1,
                             &alpha, nn->d_weights2, HIDDEN_SIZE2, nn->d_fc1_output, 1, &beta, nn->d_fc2_output, 1));
    
    // Add bias and ReLU
    bias_add_kernel_single<<<(HIDDEN_SIZE2 + 255) / 256, 256>>>(nn->d_fc2_output, nn->d_bias2, HIDDEN_SIZE2);
    relu_kernel_single<<<(HIDDEN_SIZE2 + 255) / 256, 256>>>(nn->d_fc2_output, HIDDEN_SIZE2);
    
    // Layer 3: HIDDEN_SIZE2 → OUTPUT_SIZE
    CUBLAS_CHECK(cublasSgemv(nn->cublas_handle, CUBLAS_OP_N, OUTPUT_SIZE, HIDDEN_SIZE2,
                             &alpha, nn->d_weights3, OUTPUT_SIZE, nn->d_fc2_output, 1, &beta, nn->d_fc3_output, 1));
    
    // Add bias
    bias_add_kernel_single<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->d_fc3_output, nn->d_bias3, OUTPUT_SIZE);
    
    // Copy output to host
    CUDA_CHECK(cudaMemcpy(nn->h_output, nn->d_fc3_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Find prediction
    int prediction = 0;
    float max_prob = nn->h_output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (nn->h_output[i] > max_prob) {
            max_prob = nn->h_output[i];
            prediction = i;
        }
    }
    
    return prediction;
}

void normalize_image(float *image, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        image[i] = (image[i] - mean) / std;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        printf("Usage: %s <image_file.bin> [weights_file.bin]\n", argv[0]);
        printf("Image file should contain 784 float values (28x28 flattened image)\n");
        printf("Weights file is optional, defaults to 'trained_model_weights.bin'\n");
        printf("Example: %s test_image.bin\n", argv[0]);
        printf("Example: %s test_image.bin retrained_model_best.bin\n", argv[0]);
        return 1;
    }
    
    InferenceNetwork nn;
    initialize_inference_network(&nn);
    
    // Load trained weights - use provided weights file or default
    const char* weights_file = (argc == 3) ? argv[2] : "trained_model_weights.bin";
    printf("Loading weights from: %s\n", weights_file);
    load_weights_for_inference(&nn, weights_file);
    
    // Load and normalize input image
    float *input_image = (float*)malloc(INPUT_SIZE * sizeof(float));
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s\n", argv[1]);
        return 1;
    }
    fread(input_image, sizeof(float), INPUT_SIZE, f);
    fclose(f);
    
    // Normalize the image (same as training)
    normalize_image(input_image, INPUT_SIZE);
    
    // Make prediction
    int prediction = predict(&nn, input_image);
    
    printf("Prediction: %d\n", prediction);
    printf("Confidence scores:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("  Digit %d: %.4f\n", i, nn.h_output[i]);
    }
    
    free_inference_network(&nn);
    free(input_image);
    
    return 0;
}
