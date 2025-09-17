#include "neuralNetwork.cuh"

std::map<std::string, double> Timer::timings;

float get_learning_rate(int epoch) {
    // Cosine annealing with warm restarts
    float lr = INITIAL_LEARNING_RATE * 0.5f * (1.0f + cosf(epoch * M_PI / 25.0f));
    return fmaxf(lr, MIN_LEARNING_RATE);
}


    
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

__global__ void add_noise_kernel(float *input, float *output, float *noise_std,
    int batch_size, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * total_size) return;

    int b = idx / total_size;

    // Generate random noise using cuRAND
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    float noise = curand_normal(&state) * noise_std[b];

    output[idx] = input[idx] + noise;
    output[idx] = fmaxf(0.0f, fminf(1.0f, output[idx])); // Clamp to [0,1]
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
        // Apply ReLU
        x[idx] = fmaxf(0.0f, x[idx]);
        
        if (DROPOUT_RATE > 0.0f) {
            // Use cuRAND for random number generation
            curandState state;
            curand_init(clock64(), idx, 0, &state);
            float random_val = curand_uniform(&state);
            
            // Drop neurons with probability dropout_rate
            if (random_val < DROPOUT_RATE) {
                x[idx] = 0.0f;
            } else {
                // Scale up the remaining neurons by 1/(1-dropout_rate) to maintain expected value
                x[idx] *= (1.0f / (1.0f - DROPOUT_RATE));
            }
        }
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

// HOST FUNCTION TO APPLY NOISE AUGMENTATION
void add_noise_to_batch(float *batch_input, int batch_size, float noise_level = 0.02f) {
    // Allocate noise parameters on host
    float *h_noise_std = (float *)malloc(batch_size * sizeof(float));
    
    // Set noise level for each batch item (can be different for each)
    for (int i = 0; i < batch_size; i++) {
        h_noise_std[i] = noise_level;
    }
    
    // Allocate device memory
    float *d_noise_std;
    CUDA_CHECK(cudaMalloc(&d_noise_std, batch_size * sizeof(float)));
    
    // Copy noise parameters to device
    CUDA_CHECK(cudaMemcpy(d_noise_std, h_noise_std, batch_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int total_size = INPUT_SIZE; // 784 for MNIST
    int grid_size = (batch_size * total_size + 255) / 256;
    add_noise_kernel<<<grid_size, 256>>>(batch_input, batch_input, d_noise_std, batch_size, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_noise_std));
    free(h_noise_std);
}

// FORWARD PASS ONLY - separate function with timing
void forward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Forward matmul 1: input * weights1 -> fc1_output
    {
        Timer timer("fwd_matmul1");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               HIDDEN_SIZE1, batch_size, INPUT_SIZE,
                               &alpha, nn->d_weights1, HIDDEN_SIZE1,
                               nn->d_input_batch, INPUT_SIZE, &beta,
                               nn->d_fc1_output, HIDDEN_SIZE1));
    }

    // Forward bias add 1
    {
        Timer timer("fwd_bias1");
        int total_hidden1 = batch_size * HIDDEN_SIZE1;
        int grid_hidden1 = (total_hidden1 + 255) / 256;
        bias_add_kernel<<<grid_hidden1, 256>>>(nn->d_fc1_output, nn->d_bias1, batch_size, HIDDEN_SIZE1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward ReLU 1
    {
        Timer timer("fwd_relu1");
        int total_hidden1 = batch_size * HIDDEN_SIZE1;
        int grid_hidden1 = (total_hidden1 + 255) / 256;
        relu_kernel<<<grid_hidden1, 256>>>(nn->d_fc1_output, total_hidden1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward matmul 2: fc1_output * weights2 -> fc2_output
    {
        Timer timer("fwd_matmul2");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               HIDDEN_SIZE2, batch_size, HIDDEN_SIZE1,
                               &alpha, nn->d_weights2, HIDDEN_SIZE2,
                               nn->d_fc1_output, HIDDEN_SIZE1, &beta,
                               nn->d_fc2_output, HIDDEN_SIZE2));
    }

    // Forward bias add 2
    {
        Timer timer("fwd_bias2");
        int total_hidden2 = batch_size * HIDDEN_SIZE2;
        int grid_hidden2 = (total_hidden2 + 255) / 256;
        bias_add_kernel<<<grid_hidden2, 256>>>(nn->d_fc2_output, nn->d_bias2, batch_size, HIDDEN_SIZE2);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward ReLU 2
    {
        Timer timer("fwd_relu2");
        int total_hidden2 = batch_size * HIDDEN_SIZE2;
        int grid_hidden2 = (total_hidden2 + 255) / 256;
        relu_kernel<<<grid_hidden2, 256>>>(nn->d_fc2_output, total_hidden2);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward matmul 3: fc2_output * weights3 -> fc3_output
    {
        Timer timer("fwd_matmul3");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               OUTPUT_SIZE, batch_size, HIDDEN_SIZE2,
                               &alpha, nn->d_weights3, OUTPUT_SIZE,
                               nn->d_fc2_output, HIDDEN_SIZE2, &beta,
                               nn->d_fc3_output, OUTPUT_SIZE));
    }

    // Forward bias add 3 + SYNC (only because CPU needs this data)
    {
        Timer timer("fwd_bias3");
        int total_out = batch_size * OUTPUT_SIZE;
        int grid_out = (total_out + 255) / 256;
        bias_add_kernel<<<grid_out, 256>>>(nn->d_fc3_output, nn->d_bias3, batch_size, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize()); // Required for CPU copy
    }
}

// BACKWARD PASS ONLY - separate function with timing
void backward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Zero gradients (async)
    {
        Timer timer("zero_gradients");
        CUDA_CHECK(cudaMemset(nn->d_grad_weights1, 0, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn->d_grad_weights2, 0, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn->d_grad_weights3, 0, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn->d_grad_bias1, 0, HIDDEN_SIZE1 * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn->d_grad_bias2, 0, HIDDEN_SIZE2 * sizeof(float)));
        CUDA_CHECK(cudaMemset(nn->d_grad_bias3, 0, OUTPUT_SIZE * sizeof(float)));
    }

    // Backward matmul 3a: weights3 gradients
    {
        Timer timer("bwd_matmul3");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                               OUTPUT_SIZE, HIDDEN_SIZE2, batch_size,
                               &alpha, nn->d_grad_output, OUTPUT_SIZE,
                               nn->d_fc2_output, HIDDEN_SIZE2, &beta,
                               nn->d_grad_weights3, OUTPUT_SIZE));
    }

    // Backward bias3 gradients
    {
        Timer timer("bwd_bias3");
        int total_out = batch_size * OUTPUT_SIZE;
        int grid_out = (total_out + 255) / 256;
        bias_backward_kernel<<<grid_out, 256>>>(nn->d_grad_output, nn->d_grad_bias3, batch_size, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Backward matmul 3b: hidden2 gradients
    {
        Timer timer("bwd_matmul3b");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               HIDDEN_SIZE2, batch_size, OUTPUT_SIZE,
                               &alpha, nn->d_weights3, OUTPUT_SIZE,
                               nn->d_grad_output, OUTPUT_SIZE, &beta,
                               nn->d_grad_hidden2, HIDDEN_SIZE2));
    }

    // Backward ReLU 2
    {
        Timer timer("bwd_relu2");
        int total_hidden2 = batch_size * HIDDEN_SIZE2;
        int grid_hidden2 = (total_hidden2 + 255) / 256;
        relu_backward_kernel<<<grid_hidden2, 256>>>(nn->d_grad_hidden2, nn->d_fc2_output, total_hidden2);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Backward matmul 2a: weights2 gradients
    {
        Timer timer("bwd_matmul2");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                               HIDDEN_SIZE2, HIDDEN_SIZE1, batch_size,
                               &alpha, nn->d_grad_hidden2, HIDDEN_SIZE2,
                               nn->d_fc1_output, HIDDEN_SIZE1, &beta,
                               nn->d_grad_weights2, HIDDEN_SIZE2));
    }

    // Backward bias2 gradients
    {
        Timer timer("bwd_bias2");
        int total_hidden2 = batch_size * HIDDEN_SIZE2;
        int grid_hidden2 = (total_hidden2 + 255) / 256;
        bias_backward_kernel<<<grid_hidden2, 256>>>(nn->d_grad_hidden2, nn->d_grad_bias2, batch_size, HIDDEN_SIZE2);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Backward matmul 2b: hidden1 gradients
    {
        Timer timer("bwd_matmul2b");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               HIDDEN_SIZE1, batch_size, HIDDEN_SIZE2,
                               &alpha, nn->d_weights2, HIDDEN_SIZE2,
                               nn->d_grad_hidden2, HIDDEN_SIZE2, &beta,
                               nn->d_grad_hidden1, HIDDEN_SIZE1));
    }

    // Backward ReLU 1
    {
        Timer timer("bwd_relu1");
        int total_hidden1 = batch_size * HIDDEN_SIZE1;
        int grid_hidden1 = (total_hidden1 + 255) / 256;
        relu_backward_kernel<<<grid_hidden1, 256>>>(nn->d_grad_hidden1, nn->d_fc1_output, total_hidden1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Backward matmul 1a: weights1 gradients
    {
        Timer timer("bwd_matmul1");
        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                               HIDDEN_SIZE1, INPUT_SIZE, batch_size,
                               &alpha, nn->d_grad_hidden1, HIDDEN_SIZE1,
                               nn->d_input_batch, INPUT_SIZE, &beta,
                               nn->d_grad_weights1, HIDDEN_SIZE1));
    }

    // Backward bias1 gradients
    {
        Timer timer("bwd_bias1");
        int total_hidden1 = batch_size * HIDDEN_SIZE1;
        int grid_hidden1 = (total_hidden1 + 255) / 256;
        bias_backward_kernel<<<grid_hidden1, 256>>>(nn->d_grad_hidden1, nn->d_grad_bias1, batch_size, HIDDEN_SIZE1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// WEIGHT UPDATES ONLY - separate function with timing
void update_weights_only(NeuralNetworkCUDA *nn, float lr) {
    Timer timer("weight_updates");
    float neg_lr = -lr;
    
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE1,
                           &neg_lr, nn->d_grad_weights1, 1, nn->d_weights1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE1 * HIDDEN_SIZE2,
                           &neg_lr, nn->d_grad_weights2, 1, nn->d_weights2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE2 * OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_weights3, 1, nn->d_weights3, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE1,
                           &neg_lr, nn->d_grad_bias1, 1, nn->d_bias1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE2,
                           &neg_lr, nn->d_grad_bias2, 1, nn->d_bias2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_bias3, 1, nn->d_bias3, 1));
    
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
        CUDA_CHECK(cudaMemcpy(nn->h_fc3_output, nn->d_fc3_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Compute accuracy for this batch
        int *batch_labels = labels + batch * BATCH_SIZE;
        int correct = 0;
        for (int b = 0; b < BATCH_SIZE; b++) {
            float *logits = nn->h_fc3_output + b * OUTPUT_SIZE;
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
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float));
    initialize_weights_host(h_weights1, INPUT_SIZE, HIDDEN_SIZE1);
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights1);

    float *h_weights2 = (float *)malloc(HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float));
    initialize_weights_host(h_weights2, HIDDEN_SIZE1, HIDDEN_SIZE2);
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights2);

    float *h_weights3 = (float *)malloc(HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float));
    initialize_weights_host(h_weights3, HIDDEN_SIZE2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights3, h_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights3);

    float *h_bias1 = (float *)malloc(HIDDEN_SIZE1 * sizeof(float));
    initialize_bias_host(h_bias1, HIDDEN_SIZE1);
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1, HIDDEN_SIZE1 * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias1);

    float *h_bias2 = (float *)malloc(HIDDEN_SIZE2 * sizeof(float));
    initialize_bias_host(h_bias2, HIDDEN_SIZE2);
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2, HIDDEN_SIZE2 * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias2);

    float *h_bias3 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    initialize_bias_host(h_bias3, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias3, h_bias3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias3);
}

void initialize_nn_cuda(NeuralNetworkCUDA *nn) {
    // Network weights and gradients
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias3, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias3, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc1_output, BATCH_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc2_output, BATCH_SIZE * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc3_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden1, BATCH_SIZE * HIDDEN_SIZE1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden2, BATCH_SIZE * HIDDEN_SIZE2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // PERSISTENT BUFFERS - ALLOCATED ONCE, REUSED FOR ALL BATCHES
    CUDA_CHECK(cudaMalloc(&nn->d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    nn->h_fc3_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->h_grad_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    if (!nn->h_fc3_output || !nn->h_grad_output) {
        fprintf(stderr, "Failed to allocate persistent host buffers\n");
        exit(EXIT_FAILURE);
    }

    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
    initialize_random_weights_cuda(nn);
}

void free_nn_cuda(NeuralNetworkCUDA *nn) {
    CUDA_CHECK(cudaFree(nn->d_weights1));
    CUDA_CHECK(cudaFree(nn->d_weights2));
    CUDA_CHECK(cudaFree(nn->d_weights3));
    CUDA_CHECK(cudaFree(nn->d_bias1));
    CUDA_CHECK(cudaFree(nn->d_bias2));
    CUDA_CHECK(cudaFree(nn->d_bias3));
    CUDA_CHECK(cudaFree(nn->d_grad_weights1));
    CUDA_CHECK(cudaFree(nn->d_grad_weights2));
    CUDA_CHECK(cudaFree(nn->d_grad_weights3));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias3));
    CUDA_CHECK(cudaFree(nn->d_fc1_output));
    CUDA_CHECK(cudaFree(nn->d_fc2_output));
    CUDA_CHECK(cudaFree(nn->d_fc3_output));
    CUDA_CHECK(cudaFree(nn->d_grad_hidden1));
    CUDA_CHECK(cudaFree(nn->d_grad_hidden2));
    CUDA_CHECK(cudaFree(nn->d_grad_output));
    
    // Free persistent buffers
    CUDA_CHECK(cudaFree(nn->d_input_batch));
    free(nn->h_fc3_output);
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

            // === NOISE AUGMENTATION ===
            {
                Timer timer("noise_augmentation");
                add_noise_to_batch(nn->d_input_batch, BATCH_SIZE, 0.005f); // 0.5% noise
            }

            // === FORWARD PASS ONLY ===
            forward_pass_only(nn, BATCH_SIZE);

            // === D2H Transfer (using persistent buffer) ===
            {
                Timer timer("d2h_transfer");
                CUDA_CHECK(cudaMemcpy(nn->h_fc3_output, nn->d_fc3_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            }

            // === Host Loss Computation ONLY ===
            float batch_loss = compute_loss_and_grad(BATCH_SIZE, nn->h_fc3_output, batch_labels, nn->h_grad_output);
            total_loss += batch_loss;

            // === H2D Gradient Transfer (using persistent buffer) ===
            {
                Timer timer("h2d_grad_transfer");
                CUDA_CHECK(cudaMemcpy(nn->d_grad_output, nn->h_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            }

            // === BACKWARD PASS ===
            backward_pass_only(nn, BATCH_SIZE);

            // === WEIGHT UPDATES ===
            update_weights_only(nn, get_learning_rate(epoch));
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
    //normalize_data(test_data, TEST_SIZE * INPUT_SIZE);
    load_labels("../data/y_test.bin", test_labels, TEST_SIZE);

    NeuralNetworkCUDA nn;
    initialize_nn_cuda(&nn);

    train(&nn, train_data, train_labels, test_data, test_labels);

    // Save the trained model weights
    save_weights(&nn, "./cuda/trained_model_weights.bin");
    printf("Training completed and model saved!\n");

    free_nn_cuda(&nn);
    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);

    return 0;
}