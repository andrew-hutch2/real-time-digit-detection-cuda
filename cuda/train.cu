#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include "./timer.h"

using namespace std;


std::map<std::string, double> Timer::timings;

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define HIDDEN_SIZE1 512
#define HIDDEN_SIZE2 256
#define HIDDEN_SIZE3 128
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
#define BATCH_SIZE 8
#define EPOCHS 20
#define LEARNING_RATE 0.01

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef struct {
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    float *d_fc1_output, *d_fc2_output, *d_grad_hidden, *d_grad_output;
    
    // PERSISTENT BUFFERS - NO MORE MALLOC/FREE PER BATCH
    float *d_input_batch;
    float *h_fc2_output;
    float *h_grad_output;
    
    cublasHandle_t cublas_handle;
} NeuralNetworkCUDA;

void load_data(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen data"); exit(EXIT_FAILURE); }
    fread(data, sizeof(float), size, f);
    fclose(f);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen labels"); exit(EXIT_FAILURE); }
    fread(labels, sizeof(int), size, f);
    fclose(f);
}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

void initialize_weights_host(float *weights, int rows, int cols) {
    float scale = sqrtf(2.0f / rows);
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

void initialize_bias_host(float *bias, int size) {
    memset(bias, 0, size * sizeof(float));
}

__global__ void bias_add_kernel(float *x, float *bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        x[idx] += bias[bias_idx];
    }
}

__global__ void relu_kernel(float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_backward_kernel(float *grad, float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

__global__ void bias_backward_kernel(float *grad_output, float *grad_bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias[bias_idx], grad_output[idx]);
    }
}

// FORWARD PASS ONLY - separate function with timing
void forward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Forward matmul 1: input * weights1
    {
        Timer timer("fwd_matmul1");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               HIDDEN_SIZE, batch_size, INPUT_SIZE,
                               &alpha, nn->d_weights1, HIDDEN_SIZE,
                               nn->d_input_batch, INPUT_SIZE, &beta,
                               nn->d_fc1_output, HIDDEN_SIZE));
    }

    // Forward bias add 1
    {
        Timer timer("fwd_bias1");
        int total_hidden = batch_size * HIDDEN_SIZE;
        int grid_hidden = (total_hidden + 255) / 256;
        bias_add_kernel<<<grid_hidden, 256>>>(nn->d_fc1_output, nn->d_bias1, batch_size, HIDDEN_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward ReLU
    {
        Timer timer("fwd_relu");
        int total_hidden = batch_size * HIDDEN_SIZE;
        int grid_hidden = (total_hidden + 255) / 256;
        relu_kernel<<<grid_hidden, 256>>>(nn->d_fc1_output, total_hidden);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward matmul 2: hidden * weights2
    {
        Timer timer("fwd_matmul2");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               OUTPUT_SIZE, batch_size, HIDDEN_SIZE,
                               &alpha, nn->d_weights2, OUTPUT_SIZE,
                               nn->d_fc1_output, HIDDEN_SIZE, &beta,
                               nn->d_fc2_output, OUTPUT_SIZE));
    }

    // Forward bias add 2 + SYNC (only because CPU needs this data)
    {
        Timer timer("fwd_bias2");
        int total_out = batch_size * OUTPUT_SIZE;
        int grid_out = (total_out + 255) / 256;
        bias_add_kernel<<<grid_out, 256>>>(nn->d_fc2_output, nn->d_bias2, batch_size, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize()); // Required for CPU copy
    }
}

// BACKWARD PASS ONLY - separate function with timing
void backward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Zero gradients (async)
    {
        Timer timer("zero_gradients");
        CUDA_CHECK(cudaMemset(nn->d_grad_weights1, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn->d_grad_weights2, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn->d_grad_bias1, 0, HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn->d_grad_bias2, 0, OUTPUT_SIZE * sizeof(float)));
    }

    // Backward matmul 2a: weights2 gradients
    {
        Timer timer("bwd_matmul2");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                               OUTPUT_SIZE, HIDDEN_SIZE, batch_size,
                               &alpha, nn->d_grad_output, OUTPUT_SIZE,
                               nn->d_fc1_output, HIDDEN_SIZE, &beta,
                               nn->d_grad_weights2, OUTPUT_SIZE));
    }

    // Backward bias2 gradients
    {
        Timer timer("bwd_bias2");
        int total_out = batch_size * OUTPUT_SIZE;
        int grid_out = (total_out + 255) / 256;
        bias_backward_kernel<<<grid_out, 256>>>(nn->d_grad_output, nn->d_grad_bias2, batch_size, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Backward matmul 2b: hidden gradients
    {
        Timer timer("bwd_matmul2b");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
                               &alpha, nn->d_weights2, OUTPUT_SIZE,
                               nn->d_grad_output, OUTPUT_SIZE, &beta,
                               nn->d_grad_hidden, HIDDEN_SIZE));
    }

    // Backward ReLU
    {
        Timer timer("bwd_relu");
        int total_hidden = batch_size * HIDDEN_SIZE;
        int grid_hidden = (total_hidden + 255) / 256;
        relu_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden, nn->d_fc1_output, total_hidden);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Backward matmul 1a: weights1 gradients
    {
        Timer timer("bwd_matmul1");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                               HIDDEN_SIZE, INPUT_SIZE, batch_size,
                               &alpha, nn->d_grad_hidden, HIDDEN_SIZE,
                               nn->d_input_batch, INPUT_SIZE, &beta,
                               nn->d_grad_weights1, HIDDEN_SIZE));
    }

    // Backward bias1 gradients
    {
        Timer timer("bwd_bias1");
        int total_hidden = batch_size * HIDDEN_SIZE;
        int grid_hidden = (total_hidden + 255) / 256;
        bias_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden, nn->d_grad_bias1, batch_size, HIDDEN_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// WEIGHT UPDATES ONLY - separate function with timing
void update_weights_only(NeuralNetworkCUDA *nn, float lr) {
    Timer timer("weight_updates");
    float neg_lr = -lr;
    
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_weights1, 1, nn->d_weights1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE * OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_weights2, 1, nn->d_weights2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_bias1, 1, nn->d_bias1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_bias2, 1, nn->d_bias2, 1));
    
    // Final sync only at end of entire batch (required before next iteration)
    CUDA_CHECK(cudaDeviceSynchronize());
}

float compute_loss_and_grad(int batch_size, float *h_logits, int *labels, float *h_grad) {
    Timer timer("loss_computation");
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        float *logits = h_logits + b * OUTPUT_SIZE;
        int label = labels[b];
        float max_logit = -INFINITY;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float shifted = logits[i] - max_logit;
            float expv = expf(shifted);
            sum_exp += expv;
            h_grad[b * OUTPUT_SIZE + i] = expv;
        }
        loss -= (logits[label] - max_logit - logf(sum_exp));
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            h_grad[b * OUTPUT_SIZE + i] /= sum_exp;
        }
        h_grad[b * OUTPUT_SIZE + label] -= 1.0f;
    }
    for (int i = 0; i < batch_size * OUTPUT_SIZE; i++) {
        h_grad[i] /= batch_size;
    }
    return loss / batch_size;
}

// Function to compute accuracy
float compute_accuracy(int batch_size, float *h_logits, int *labels) {
    int correct = 0;
    for (int b = 0; b < batch_size; b++) {
        float *logits = h_logits + b * OUTPUT_SIZE;
        int label = labels[b];
        
        // Find the predicted class (argmax)
        int predicted = 0;
        float max_logit = logits[0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                predicted = i;
            }
        }
        
        if (predicted == label) {
            correct++;
        }
    }
    return (float)correct / batch_size;
}

// Function to evaluate accuracy on a dataset
float evaluate_accuracy(NeuralNetworkCUDA *nn, float *data, int *labels, int dataset_size) {
    int num_batches = dataset_size / BATCH_SIZE;
    int total_correct = 0;
    int total_samples = 0;
    
    for (int batch = 0; batch < num_batches; batch++) {
        float *batch_input = data + batch * BATCH_SIZE * INPUT_SIZE;
        
        // Copy input to GPU
        CUDA_CHECK(cudaMemcpy(nn->d_input_batch, batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass_only(nn, BATCH_SIZE);
        
        // Copy output back to CPU
        CUDA_CHECK(cudaMemcpy(nn->h_fc2_output, nn->d_fc2_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Compute accuracy for this batch
        int *batch_labels = labels + batch * BATCH_SIZE;
        int correct = 0;
        for (int b = 0; b < BATCH_SIZE; b++) {
            float *logits = nn->h_fc2_output + b * OUTPUT_SIZE;
            int label = batch_labels[b];
            
            // Find the predicted class (argmax)
            int predicted = 0;
            float max_logit = logits[0];
            for (int i = 1; i < OUTPUT_SIZE; i++) {
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                    predicted = i;
                }
            }
            
            if (predicted == label) {
                correct++;
            }
        }
        
        total_correct += correct;
        total_samples += BATCH_SIZE;
    }
    
    return (float)total_correct / total_samples;
}

void initialize_random_weights_cuda(NeuralNetworkCUDA *nn) {
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    initialize_weights_host(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights1);

    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    initialize_weights_host(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights2);

    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    initialize_bias_host(h_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias1);

    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    initialize_bias_host(h_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias2);
}

void initialize_nn_cuda(NeuralNetworkCUDA *nn) {
    // Network weights and gradients
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc1_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc2_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // PERSISTENT BUFFERS - ALLOCATED ONCE, REUSED FOR ALL BATCHES
    CUDA_CHECK(cudaMalloc(&nn->d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    nn->h_fc2_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->h_grad_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    if (!nn->h_fc2_output || !nn->h_grad_output) {
        fprintf(stderr, "Failed to allocate persistent host buffers\n");
        exit(EXIT_FAILURE);
    }

    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
    initialize_random_weights_cuda(nn);
}

void free_nn_cuda(NeuralNetworkCUDA *nn) {
    CUDA_CHECK(cudaFree(nn->d_weights1));
    CUDA_CHECK(cudaFree(nn->d_weights2));
    CUDA_CHECK(cudaFree(nn->d_bias1));
    CUDA_CHECK(cudaFree(nn->d_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_weights1));
    CUDA_CHECK(cudaFree(nn->d_grad_weights2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2));
    CUDA_CHECK(cudaFree(nn->d_fc1_output));
    CUDA_CHECK(cudaFree(nn->d_fc2_output));
    CUDA_CHECK(cudaFree(nn->d_grad_hidden));
    CUDA_CHECK(cudaFree(nn->d_grad_output));
    
    // Free persistent buffers
    CUDA_CHECK(cudaFree(nn->d_input_batch));
    free(nn->h_fc2_output);
    free(nn->h_grad_output);
    
    CUBLAS_CHECK(cublasDestroy(nn->cublas_handle));
}

// Train function with automatic timing
void train(NeuralNetworkCUDA *nn, float *X_train, int *y_train, float *X_test, int *y_test) {
    int num_batches = TRAIN_SIZE / BATCH_SIZE;
    
    // Reset timing stats
    Timer::reset();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        for (int batch = 0; batch < num_batches; batch++) {
            float *batch_input = X_train + batch * BATCH_SIZE * INPUT_SIZE;
            int *batch_labels = y_train + batch * BATCH_SIZE;

            // === H2D Transfer (using persistent buffer) ===
            {
                Timer timer("h2d_transfer");
                CUDA_CHECK(cudaMemcpy(nn->d_input_batch, batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            }

            // === FORWARD PASS ONLY ===
            forward_pass_only(nn, BATCH_SIZE);

            // === D2H Transfer (using persistent buffer) ===
            {
                Timer timer("d2h_transfer");
                CUDA_CHECK(cudaMemcpy(nn->h_fc2_output, nn->d_fc2_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            }

            // === Host Loss Computation ONLY ===
            float batch_loss = compute_loss_and_grad(BATCH_SIZE, nn->h_fc2_output, batch_labels, nn->h_grad_output);
            total_loss += batch_loss;

            // === H2D Gradient Transfer (using persistent buffer) ===
            {
                Timer timer("h2d_grad_transfer");
                CUDA_CHECK(cudaMemcpy(nn->d_grad_output, nn->h_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            }

            // === BACKWARD PASS ===
            backward_pass_only(nn, BATCH_SIZE);

            // === WEIGHT UPDATES ===
            update_weights_only(nn, LEARNING_RATE);
        }
        
        // Calculate accuracies after each epoch
        float train_acc = evaluate_accuracy(nn, X_train, y_train, TRAIN_SIZE);
        float test_acc = evaluate_accuracy(nn, X_test, y_test, TEST_SIZE);
        
        printf("Epoch %d - Loss: %.4f, Train Acc: %.2f%%, Test Acc: %.2f%%\n", 
               epoch, total_loss / num_batches, train_acc * 100.0f, test_acc * 100.0f);
    }
    
    // Print timing breakdown
    Timer::print_timings();
}

int main() {
    srand(12345); // Fixed seed for debugging

    float *train_data = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *train_labels = (int *)malloc(TRAIN_SIZE * sizeof(int));
    float *test_data = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *test_labels = (int *)malloc(TEST_SIZE * sizeof(int));
    
    load_data("../data/X_train.bin", train_data, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(train_data, TRAIN_SIZE * INPUT_SIZE);
    load_labels("../data/y_train.bin", train_labels, TRAIN_SIZE);
    
    load_data("../data/X_test.bin", test_data, TEST_SIZE * INPUT_SIZE);
    normalize_data(test_data, TEST_SIZE * INPUT_SIZE);
    load_labels("../data/y_test.bin", test_labels, TEST_SIZE);

    NeuralNetworkCUDA nn;
    initialize_nn_cuda(&nn);

    train(&nn, train_data, train_labels, test_data, test_labels);

    free_nn_cuda(&nn);
    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);

    return 0;
}