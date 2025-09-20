#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <ctime>
#include "./timer.h"

// Network architecture constants
#define INPUT_SIZE 784
#define HIDDEN_SIZE1 1568
#define HIDDEN_SIZE2 784
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 20000
#define TEST_SIZE 5000
#define BATCH_SIZE 32
#define EPOCHS 100
#define INITIAL_LEARNING_RATE .01
#define MIN_LEARNING_RATE 0.0001
#define DROPOUT_RATE 0.2
#define WEIGHT_DECAY 0.0001

// CUDA error checking macros
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

using namespace std;

typedef struct {
    // Weights for 2 hidden layers
    float *d_weights1;  // INPUT_SIZE → HIDDEN_SIZE1 (784 → 512)
    float *d_weights2;  // HIDDEN_SIZE1 → HIDDEN_SIZE2 (512 → 256) 
    float *d_weights3;  // HIDDEN_SIZE2 → OUTPUT_SIZE (256 → 10)
    
    // Biases for 2 hidden layers
    float *d_bias1;     // HIDDEN_SIZE1 (512)
    float *d_bias2;     // HIDDEN_SIZE2 (256)
    float *d_bias3;     // OUTPUT_SIZE (10)
    
    // Gradients for weights
    float *d_grad_weights1;  // INPUT_SIZE × HIDDEN_SIZE1
    float *d_grad_weights2;  // HIDDEN_SIZE1 × HIDDEN_SIZE2
    float *d_grad_weights3;  // HIDDEN_SIZE2 × OUTPUT_SIZE
    
    // Gradients for biases
    float *d_grad_bias1;     // HIDDEN_SIZE1
    float *d_grad_bias2;     // HIDDEN_SIZE2
    float *d_grad_bias3;     // OUTPUT_SIZE
    
    // Layer outputs (activations)
    float *d_fc1_output;     // BATCH_SIZE × HIDDEN_SIZE1
    float *d_fc2_output;     // BATCH_SIZE × HIDDEN_SIZE2
    float *d_fc3_output;     // BATCH_SIZE × OUTPUT_SIZE
    
    // Gradients for layer outputs
    float *d_grad_hidden1;   // BATCH_SIZE × HIDDEN_SIZE1
    float *d_grad_hidden2;   // BATCH_SIZE × HIDDEN_SIZE2
    float *d_grad_output;    // BATCH_SIZE × OUTPUT_SIZE
    
    // PERSISTENT BUFFERS
    float *d_input_batch;
    float *h_fc3_output;     // Changed from h_fc4_output
    float *h_grad_output;
    
    cublasHandle_t cublas_handle;
} NeuralNetworkCUDA;

// Function declarations
float get_learning_rate(int epoch);
void load_data(const char *filename, float *data, int size);
void load_labels(const char *filename, int *labels, int size);
void normalize_data(float *data, int size);
void initialize_weights_host(float *weights, int rows, int cols);
void initialize_bias_host(float *bias, int size);
void save_weights(NeuralNetworkCUDA *nn, const char *filename);
void load_weights(NeuralNetworkCUDA *nn, const char *filename);

// Neural network management functions
void initialize_network(NeuralNetworkCUDA *nn);
void free_network(NeuralNetworkCUDA *nn);

// Forward and backward pass functions
void forward_pass(NeuralNetworkCUDA *nn, float *d_input, float *d_output);
void forward_pass_batch(NeuralNetworkCUDA *nn, float *d_images, float *d_outputs, int batch_size);
void backward_pass_batch(NeuralNetworkCUDA *nn, float *d_images, float *d_outputs, float *d_targets, int batch_size, float learning_rate);

// Loss computation functions
float compute_cross_entropy_loss(float *d_outputs, float *d_targets, int batch_size);
float compute_loss_and_grad(int batch_size, float *h_logits, int *labels, float *h_grad);
void update_weights(NeuralNetworkCUDA *nn, float learning_rate);

// CUDA kernel declarations
__global__ void add_noise_kernel(float *input, float *output, float *noise_std, int batch_size, int total_size);
__global__ void bias_add_kernel(float *x, float *bias, int batch, int size);
__global__ void relu_kernel(float *x, int total);
__global__ void relu_backward_kernel(float *x, float *grad, int total);
__global__ void bias_backward_kernel(float *grad, float *grad_bias, int batch, int size);
__global__ void create_one_hot_targets(float *d_targets, int *d_labels, int batch_size, int num_classes);

#endif // NEURAL_NETWORK_CUH