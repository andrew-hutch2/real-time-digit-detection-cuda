#include "neuralNetwork.cuh"

// === WEIGHT SAVING/LOADING FUNCTIONS ===
void save_weights(NeuralNetworkCUDA *nn, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return;
    }
    
    // Create temporary host buffers for weights and biases
    float *h_weights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float));
    float *h_weights2 = (float*)malloc(HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float));
    float *h_weights3 = (float*)malloc(HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float));
    float *h_bias1 = (float*)malloc(HIDDEN_SIZE1 * sizeof(float));
    float *h_bias2 = (float*)malloc(HIDDEN_SIZE2 * sizeof(float));
    float *h_bias3 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Copy weights from device to host
    CUDA_CHECK(cudaMemcpy(h_weights1, nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_weights2, nn->d_weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_weights3, nn->d_weights3, HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias1, nn->d_bias1, HIDDEN_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias2, nn->d_bias2, HIDDEN_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias3, nn->d_bias3, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write network architecture info
    int input_size = INPUT_SIZE;
    int hidden_size1 = HIDDEN_SIZE1;
    int hidden_size2 = HIDDEN_SIZE2;
    int output_size = OUTPUT_SIZE;
    fwrite(&input_size, sizeof(int), 1, f);
    fwrite(&hidden_size1, sizeof(int), 1, f);
    fwrite(&hidden_size2, sizeof(int), 1, f);
    fwrite(&output_size, sizeof(int), 1, f);
    
    // Write weights and biases
    fwrite(h_weights1, sizeof(float), INPUT_SIZE * HIDDEN_SIZE1, f);
    fwrite(h_weights2, sizeof(float), HIDDEN_SIZE1 * HIDDEN_SIZE2, f);
    fwrite(h_weights3, sizeof(float), HIDDEN_SIZE2 * OUTPUT_SIZE, f);
    fwrite(h_bias1, sizeof(float), HIDDEN_SIZE1, f);
    fwrite(h_bias2, sizeof(float), HIDDEN_SIZE2, f);
    fwrite(h_bias3, sizeof(float), OUTPUT_SIZE, f);
    
    fclose(f);
    
    // Free temporary buffers
    free(h_weights1);
    free(h_weights2);
    free(h_weights3);
    free(h_bias1);
    free(h_bias2);
    free(h_bias3);
    
    printf("Model weights saved to %s\n", filename);
}

void load_weights(NeuralNetworkCUDA *nn, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for reading\n", filename);
        return;
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
        return;
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
