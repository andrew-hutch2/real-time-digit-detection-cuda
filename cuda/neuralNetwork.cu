#include "neuralNetwork.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// CUDA kernels
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

__global__ void relu_backward_kernel(float *x, float *grad, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad[idx] *= (x[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void bias_backward_kernel(float *grad, float *grad_bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias[bias_idx], grad[idx]);
    }
}

__global__ void add_noise_kernel(float *input, float *output, float *noise_std, int batch_size, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * total_size) {
        // Simple noise generation using thread ID for pseudo-randomness
        unsigned int seed = idx + blockIdx.x * blockDim.x;
        float noise = ((float)(seed % 1000) / 1000.0f - 0.5f) * 2.0f * noise_std[0];
        output[idx] = input[idx] + noise;
    }
}

// Host functions
void initialize_weights_host(float *weights, int rows, int cols) {
    float scale = sqrtf(2.0f / (rows + cols));
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

void initialize_bias_host(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// CUDA kernel for creating one-hot encoded targets
__global__ void create_one_hot_targets(float *d_targets, int *d_labels, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int label = d_labels[idx];
        for (int i = 0; i < num_classes; i++) {
            d_targets[idx * num_classes + i] = (i == label) ? 1.0f : 0.0f;
        }
    }
}

// Initialize neural network
void initialize_network(NeuralNetworkCUDA *nn) {
    // Initialize cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
    
    // Allocate device memory for weights
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate device memory for biases
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias3, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate device memory for gradients
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias3, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate device memory for layer outputs
    CUDA_CHECK(cudaMalloc(&nn->d_fc1_output, BATCH_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc2_output, BATCH_SIZE * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc3_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate device memory for gradients
    CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden1, BATCH_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden2, BATCH_SIZE * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate persistent buffers
    CUDA_CHECK(cudaMalloc(&nn->d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    nn->h_fc3_output = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->h_grad_output = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    
    // Initialize weights with Xavier initialization
    float *h_weights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float));
    float *h_weights2 = (float*)malloc(HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float));
    float *h_weights3 = (float*)malloc(HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float));
    
    initialize_weights_host(h_weights1, INPUT_SIZE, HIDDEN_SIZE1);
    initialize_weights_host(h_weights2, HIDDEN_SIZE1, HIDDEN_SIZE2);
    initialize_weights_host(h_weights3, HIDDEN_SIZE2, OUTPUT_SIZE);
    
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->d_weights3, h_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize biases to zero
    CUDA_CHECK(cudaMemset(nn->d_bias1, 0, HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_bias2, 0, HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_bias3, 0, OUTPUT_SIZE * sizeof(float)));
    
    free(h_weights1);
    free(h_weights2);
    free(h_weights3);
}

// Free neural network memory
void free_network(NeuralNetworkCUDA *nn) {
    if (nn->cublas_handle) {
        cublasDestroy(nn->cublas_handle);
    }
    
    if (nn->d_weights1) cudaFree(nn->d_weights1);
    if (nn->d_weights2) cudaFree(nn->d_weights2);
    if (nn->d_weights3) cudaFree(nn->d_weights3);
    if (nn->d_bias1) cudaFree(nn->d_bias1);
    if (nn->d_bias2) cudaFree(nn->d_bias2);
    if (nn->d_bias3) cudaFree(nn->d_bias3);
    if (nn->d_grad_weights1) cudaFree(nn->d_grad_weights1);
    if (nn->d_grad_weights2) cudaFree(nn->d_grad_weights2);
    if (nn->d_grad_weights3) cudaFree(nn->d_grad_weights3);
    if (nn->d_grad_bias1) cudaFree(nn->d_grad_bias1);
    if (nn->d_grad_bias2) cudaFree(nn->d_grad_bias2);
    if (nn->d_grad_bias3) cudaFree(nn->d_grad_bias3);
    if (nn->d_fc1_output) cudaFree(nn->d_fc1_output);
    if (nn->d_fc2_output) cudaFree(nn->d_fc2_output);
    if (nn->d_fc3_output) cudaFree(nn->d_fc3_output);
    if (nn->d_grad_hidden1) cudaFree(nn->d_grad_hidden1);
    if (nn->d_grad_hidden2) cudaFree(nn->d_grad_hidden2);
    if (nn->d_grad_output) cudaFree(nn->d_grad_output);
    if (nn->d_input_batch) cudaFree(nn->d_input_batch);
    if (nn->h_fc3_output) free(nn->h_fc3_output);
    if (nn->h_grad_output) free(nn->h_grad_output);
}

// Forward pass for single sample
void forward_pass(NeuralNetworkCUDA *nn, float *d_input, float *d_output) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // First layer: input -> hidden1
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           HIDDEN_SIZE1, 1, INPUT_SIZE,
                           &alpha, nn->d_weights1, HIDDEN_SIZE1,
                           d_input, INPUT_SIZE, &beta,
                           nn->d_fc1_output, HIDDEN_SIZE1));
    
    // Add bias and ReLU
    bias_add_kernel<<<(HIDDEN_SIZE1 + 255) / 256, 256>>>(nn->d_fc1_output, nn->d_bias1, 1, HIDDEN_SIZE1);
    relu_kernel<<<(HIDDEN_SIZE1 + 255) / 256, 256>>>(nn->d_fc1_output, HIDDEN_SIZE1);
    
    // Second layer: hidden1 -> hidden2
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           HIDDEN_SIZE2, 1, HIDDEN_SIZE1,
                           &alpha, nn->d_weights2, HIDDEN_SIZE2,
                           nn->d_fc1_output, HIDDEN_SIZE1, &beta,
                           nn->d_fc2_output, HIDDEN_SIZE2));
    
    // Add bias and ReLU
    bias_add_kernel<<<(HIDDEN_SIZE2 + 255) / 256, 256>>>(nn->d_fc2_output, nn->d_bias2, 1, HIDDEN_SIZE2);
    relu_kernel<<<(HIDDEN_SIZE2 + 255) / 256, 256>>>(nn->d_fc2_output, HIDDEN_SIZE2);
    
    // Third layer: hidden2 -> output
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           OUTPUT_SIZE, 1, HIDDEN_SIZE2,
                           &alpha, nn->d_weights3, OUTPUT_SIZE,
                           nn->d_fc2_output, HIDDEN_SIZE2, &beta,
                           d_output, OUTPUT_SIZE));
    
    // Add bias (no activation for output layer)
    bias_add_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->d_bias3, 1, OUTPUT_SIZE);
}

// Forward pass for batch
void forward_pass_batch(NeuralNetworkCUDA *nn, float *d_images, float *d_outputs, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // First layer: input -> hidden1
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           HIDDEN_SIZE1, batch_size, INPUT_SIZE,
                           &alpha, nn->d_weights1, HIDDEN_SIZE1,
                           d_images, INPUT_SIZE, &beta,
                           nn->d_fc1_output, HIDDEN_SIZE1));
    
    // Add bias and ReLU
    bias_add_kernel<<<(HIDDEN_SIZE1 * batch_size + 255) / 256, 256>>>(nn->d_fc1_output, nn->d_bias1, batch_size, HIDDEN_SIZE1);
    relu_kernel<<<(HIDDEN_SIZE1 * batch_size + 255) / 256, 256>>>(nn->d_fc1_output, HIDDEN_SIZE1 * batch_size);
    
    // Second layer: hidden1 -> hidden2
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           HIDDEN_SIZE2, batch_size, HIDDEN_SIZE1,
                           &alpha, nn->d_weights2, HIDDEN_SIZE2,
                           nn->d_fc1_output, HIDDEN_SIZE1, &beta,
                           nn->d_fc2_output, HIDDEN_SIZE2));
    
    // Add bias and ReLU
    bias_add_kernel<<<(HIDDEN_SIZE2 * batch_size + 255) / 256, 256>>>(nn->d_fc2_output, nn->d_bias2, batch_size, HIDDEN_SIZE2);
    relu_kernel<<<(HIDDEN_SIZE2 * batch_size + 255) / 256, 256>>>(nn->d_fc2_output, HIDDEN_SIZE2 * batch_size);
    
    // Third layer: hidden2 -> output
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           OUTPUT_SIZE, batch_size, HIDDEN_SIZE2,
                           &alpha, nn->d_weights3, OUTPUT_SIZE,
                           nn->d_fc2_output, HIDDEN_SIZE2, &beta,
                           d_outputs, OUTPUT_SIZE));
    
    // Add bias and apply softmax for output layer
    bias_add_kernel<<<(OUTPUT_SIZE * batch_size + 255) / 256, 256>>>(d_outputs, nn->d_bias3, batch_size, OUTPUT_SIZE);
    
    // Apply softmax to outputs
    // Note: This is a simplified softmax - in practice you'd want a more numerically stable version
    // For now, we'll skip softmax and let the loss function handle it
}

// Compute loss and gradients on host (based on train2.cu)
float compute_loss_and_grad(int batch_size, float *h_logits, int *labels, float *h_grad) {
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

// Backward pass for batch (based on train2.cu implementation)
void backward_pass_batch(NeuralNetworkCUDA *nn, float *d_images, float *d_outputs, float *d_targets, int batch_size, float learning_rate) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Zero gradients
    CUDA_CHECK(cudaMemset(nn->d_grad_weights1, 0, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_weights2, 0, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_weights3, 0, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias1, 0, HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias2, 0, HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias3, 0, OUTPUT_SIZE * sizeof(float)));
    
    // Backward matmul 3a: weights3 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           OUTPUT_SIZE, HIDDEN_SIZE2, batch_size,
                           &alpha, nn->d_grad_output, OUTPUT_SIZE,
                           nn->d_fc2_output, HIDDEN_SIZE2, &beta,
                           nn->d_grad_weights3, OUTPUT_SIZE));
    
    // Backward bias3 gradients
    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_backward_kernel<<<grid_out, 256>>>(nn->d_grad_output, nn->d_grad_bias3, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Backward matmul 3b: hidden2 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           HIDDEN_SIZE2, batch_size, OUTPUT_SIZE,
                           &alpha, nn->d_weights3, OUTPUT_SIZE,
                           nn->d_grad_output, OUTPUT_SIZE, &beta,
                           nn->d_grad_hidden2, HIDDEN_SIZE2));
    
    // Backward ReLU 2
    int total_hidden2 = batch_size * HIDDEN_SIZE2;
    int grid_hidden2 = (total_hidden2 + 255) / 256;
    relu_backward_kernel<<<grid_hidden2, 256>>>(nn->d_grad_hidden2, nn->d_fc2_output, total_hidden2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Backward matmul 2a: weights2 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           HIDDEN_SIZE2, HIDDEN_SIZE1, batch_size,
                           &alpha, nn->d_grad_hidden2, HIDDEN_SIZE2,
                           nn->d_fc1_output, HIDDEN_SIZE1, &beta,
                           nn->d_grad_weights2, HIDDEN_SIZE2));
    
    // Backward bias2 gradients
    bias_backward_kernel<<<grid_hidden2, 256>>>(nn->d_grad_hidden2, nn->d_grad_bias2, batch_size, HIDDEN_SIZE2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Backward matmul 2b: hidden1 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           HIDDEN_SIZE1, batch_size, HIDDEN_SIZE2,
                           &alpha, nn->d_weights2, HIDDEN_SIZE2,
                           nn->d_grad_hidden2, HIDDEN_SIZE2, &beta,
                           nn->d_grad_hidden1, HIDDEN_SIZE1));
    
    // Backward ReLU 1
    int total_hidden1 = batch_size * HIDDEN_SIZE1;
    int grid_hidden1 = (total_hidden1 + 255) / 256;
    relu_backward_kernel<<<grid_hidden1, 256>>>(nn->d_grad_hidden1, nn->d_fc1_output, total_hidden1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Backward matmul 1a: weights1 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           HIDDEN_SIZE1, INPUT_SIZE, batch_size,
                           &alpha, nn->d_grad_hidden1, HIDDEN_SIZE1,
                           d_images, INPUT_SIZE, &beta,
                           nn->d_grad_weights1, HIDDEN_SIZE1));
    
    // Backward bias1 gradients
    bias_backward_kernel<<<grid_hidden1, 256>>>(nn->d_grad_hidden1, nn->d_grad_bias1, batch_size, HIDDEN_SIZE1);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Update weights with L2 regularization (weight decay)
void update_weights(NeuralNetworkCUDA *nn, float learning_rate) {
    float neg_lr = -learning_rate;
    float weight_decay_lr = -learning_rate * WEIGHT_DECAY;
    
    // Update weights with gradients and weight decay
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE1,
                           &neg_lr, nn->d_grad_weights1, 1, nn->d_weights1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE1,
                           &weight_decay_lr, nn->d_weights1, 1, nn->d_weights1, 1));
    
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE1 * HIDDEN_SIZE2,
                           &neg_lr, nn->d_grad_weights2, 1, nn->d_weights2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE1 * HIDDEN_SIZE2,
                           &weight_decay_lr, nn->d_weights2, 1, nn->d_weights2, 1));
    
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE2 * OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_weights3, 1, nn->d_weights3, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE2 * OUTPUT_SIZE,
                           &weight_decay_lr, nn->d_weights3, 1, nn->d_weights3, 1));
    
    // Update biases (no weight decay for biases)
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE1,
                           &neg_lr, nn->d_grad_bias1, 1, nn->d_bias1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE2,
                           &neg_lr, nn->d_grad_bias2, 1, nn->d_bias2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_bias3, 1, nn->d_bias3, 1));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Compute cross-entropy loss
float compute_cross_entropy_loss(float *d_outputs, float *d_targets, int batch_size) {
    // Copy outputs to host for loss computation
    float *h_outputs = (float*)malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    float *h_targets = (float*)malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_outputs, d_outputs, batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_targets, d_targets, batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < batch_size; i++) {
        float *logits = h_outputs + i * OUTPUT_SIZE;
        float *targets = h_targets + i * OUTPUT_SIZE;
        
        // Apply softmax
        float max_logit = logits[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
            }
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum_exp += expf(logits[j] - max_logit);
        }
        
        // Compute cross-entropy loss
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (targets[j] > 0.0f) {
                float prob = expf(logits[j] - max_logit) / sum_exp;
                total_loss -= targets[j] * logf(prob + 1e-8f);
            }
        }
    }
    
    free(h_outputs);
    free(h_targets);
    
    return total_loss / batch_size;
}
