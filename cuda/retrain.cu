#include "neuralNetwork.cuh"
#include <dirent.h>
#include <sys/stat.h>

// === RETRAINING FUNCTIONS ===

typedef struct {
    float *images;
    int *labels;
    int count;
} Dataset;

Dataset load_camera_dataset(const char* data_dir) {
    Dataset dataset = {NULL, NULL, 0};
    
    // Count total samples
    int total_samples = 0;
    for (int digit = 0; digit < 10; digit++) {
        char digit_dir[256];
        snprintf(digit_dir, sizeof(digit_dir), "%s/train/%d", data_dir, digit);
        
        DIR *dir = opendir(digit_dir);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (strstr(entry->d_name, ".bin") != NULL) {
                    total_samples++;
                }
            }
            closedir(dir);
        }
    }
    
    if (total_samples == 0) {
        printf("No training samples found in %s\n", data_dir);
        return dataset;
    }
    
    printf("Found %d training samples\n", total_samples);
    
    // Allocate memory
    dataset.images = (float*)malloc(total_samples * INPUT_SIZE * sizeof(float));
    dataset.labels = (int*)malloc(total_samples * sizeof(int));
    dataset.count = total_samples;
    
    // Load samples
    int sample_idx = 0;
    for (int digit = 0; digit < 10; digit++) {
        char digit_dir[256];
        snprintf(digit_dir, sizeof(digit_dir), "%s/train/%d", data_dir, digit);
        
        DIR *dir = opendir(digit_dir);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (strstr(entry->d_name, ".bin") != NULL) {
                    char filepath[512];
                    snprintf(filepath, sizeof(filepath), "%s/%s", digit_dir, entry->d_name);
                    
                    // Load image data
                    FILE *f = fopen(filepath, "rb");
                    if (f) {
                        fread(&dataset.images[sample_idx * INPUT_SIZE], sizeof(float), INPUT_SIZE, f);
                        dataset.labels[sample_idx] = digit;
                        fclose(f);
                        sample_idx++;
                    }
                }
            }
            closedir(dir);
        }
    }
    
    printf("Loaded %d samples for retraining\n", sample_idx);
    return dataset;
}

Dataset load_validation_dataset(const char* data_dir) {
    Dataset dataset = {NULL, NULL, 0};
    
    // Count validation samples
    int total_samples = 0;
    for (int digit = 0; digit < 10; digit++) {
        char digit_dir[256];
        snprintf(digit_dir, sizeof(digit_dir), "%s/val/%d", data_dir, digit);
        
        DIR *dir = opendir(digit_dir);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (strstr(entry->d_name, ".bin") != NULL) {
                    total_samples++;
                }
            }
            closedir(dir);
        }
    }
    
    if (total_samples == 0) {
        printf("No validation samples found in %s\n", data_dir);
        return dataset;
    }
    
    printf("Found %d validation samples\n", total_samples);
    
    // Allocate memory
    dataset.images = (float*)malloc(total_samples * INPUT_SIZE * sizeof(float));
    dataset.labels = (int*)malloc(total_samples * sizeof(int));
    dataset.count = total_samples;
    
    // Load samples
    int sample_idx = 0;
    for (int digit = 0; digit < 10; digit++) {
        char digit_dir[256];
        snprintf(digit_dir, sizeof(digit_dir), "%s/val/%d", data_dir, digit);
        
        DIR *dir = opendir(digit_dir);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (strstr(entry->d_name, ".bin") != NULL) {
                    char filepath[512];
                    snprintf(filepath, sizeof(filepath), "%s/%s", digit_dir, entry->d_name);
                    
                    // Load image data
                    FILE *f = fopen(filepath, "rb");
                    if (f) {
                        fread(&dataset.images[sample_idx * INPUT_SIZE], sizeof(float), INPUT_SIZE, f);
                        dataset.labels[sample_idx] = digit;
                        fclose(f);
                        sample_idx++;
                    }
                }
            }
            closedir(dir);
        }
    }
    
    printf("Loaded %d validation samples\n", sample_idx);
    return dataset;
}

void free_dataset(Dataset *dataset) {
    if (dataset->images) free(dataset->images);
    if (dataset->labels) free(dataset->labels);
    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->count = 0;
}

float evaluate_model(NeuralNetworkCUDA *nn, Dataset *val_dataset) {
    if (val_dataset->count == 0) return 0.0f;
    
    int correct = 0;
    float *d_input = NULL;
    float *d_output = NULL;
    
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));
    
    for (int i = 0; i < val_dataset->count; i++) {
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, &val_dataset->images[i * INPUT_SIZE], 
                             INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass(nn, d_input, d_output);
        
        // Copy output to host
        float *h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Find prediction
        int prediction = 0;
        float max_prob = h_output[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > max_prob) {
                max_prob = h_output[j];
                prediction = j;
            }
        }
        
        if (prediction == val_dataset->labels[i]) {
            correct++;
        }
        
        free(h_output);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return (float)correct / val_dataset->count;
}

void retrain_model(NeuralNetworkCUDA *nn, Dataset *train_dataset, Dataset *val_dataset, 
                   int epochs, float learning_rate, const char* base_weights_file) {
    
    printf("\n=== Starting Retraining ===\n");
    printf("Training samples: %d\n", train_dataset->count);
    printf("Validation samples: %d\n", val_dataset->count);
    printf("Epochs: %d\n", epochs);
    printf("Learning rate: %f\n", learning_rate);
    printf("Base weights: %s\n", base_weights_file);
    
    // Load base weights if provided
    if (base_weights_file && strlen(base_weights_file) > 0) {
        printf("Loading base weights from %s...\n", base_weights_file);
        load_weights(nn, base_weights_file);
    }
    
    // Allocate device memory for training (batch-sized buffers)
    float *d_images = NULL;
    int *d_labels = NULL;
    float *d_outputs = NULL;
    float *d_targets = NULL;
    
    CUDA_CHECK(cudaMalloc(&d_images, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_outputs, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Training loop
    float best_val_accuracy = 0.0f;
    int patience = 10;  // Early stopping patience
    int patience_counter = 0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("\nEpoch %d/%d\n", epoch + 1, epochs);
        
        // Process data in batches
        int num_batches = (train_dataset->count + BATCH_SIZE - 1) / BATCH_SIZE;  // Ceiling division
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * BATCH_SIZE;
            int batch_size = min(BATCH_SIZE, train_dataset->count - batch_start);
            
            // Copy batch data to device
            CUDA_CHECK(cudaMemcpy(d_images, &train_dataset->images[batch_start * INPUT_SIZE], 
                                 batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels, &train_dataset->labels[batch_start], 
                                 batch_size * sizeof(int), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_batch(nn, d_images, d_outputs, batch_size);
            
            // Copy outputs to host for loss computation
            float *h_outputs = (float*)malloc(batch_size * OUTPUT_SIZE * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_outputs, d_outputs, batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Copy labels to host
            int *h_labels = (int*)malloc(batch_size * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_labels, d_labels, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
            
            // Allocate host gradient buffer
            float *h_grad = (float*)malloc(batch_size * OUTPUT_SIZE * sizeof(float));
            
            // Compute loss and gradients on host
            float batch_loss = compute_loss_and_grad(batch_size, h_outputs, h_labels, h_grad);
            total_loss += batch_loss;
            
            // Copy gradients to device
            CUDA_CHECK(cudaMemcpy(nn->d_grad_output, h_grad, batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            
            // Backward pass
            backward_pass_batch(nn, d_images, d_outputs, d_targets, batch_size, learning_rate);
            
            // Update weights
            update_weights(nn, learning_rate);
            
            // Clean up host memory
            free(h_outputs);
            free(h_labels);
            free(h_grad);
        }
        
        float loss = total_loss / num_batches;
        
        // Evaluate on validation set
        float val_accuracy = 0.0f;
        if (val_dataset->count > 0) {
            val_accuracy = evaluate_model(nn, val_dataset);
            printf("Validation accuracy: %.4f\n", val_accuracy);
            
            // Early stopping
            if (val_accuracy > best_val_accuracy) {
                best_val_accuracy = val_accuracy;
                patience_counter = 0;
                
                // Save best model
                save_weights(nn, "retrained_model_best.bin");
                printf("New best model saved!\n");
            } else {
                patience_counter++;
                if (patience_counter >= patience) {
                    printf("Early stopping triggered (patience: %d)\n", patience);
                    break;
                }
            }
        }
        
        printf("Training loss: %.6f\n", loss);
        
        // Save checkpoint every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            char checkpoint_name[256];
            snprintf(checkpoint_name, sizeof(checkpoint_name), "retrained_model_epoch_%d.bin", epoch + 1);
            save_weights(nn, checkpoint_name);
            printf("Checkpoint saved: %s\n", checkpoint_name);
        }
    }
    
    // Clean up device memory
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_outputs);
    cudaFree(d_targets);
    
    printf("\n=== Retraining Complete ===\n");
    printf("Best validation accuracy: %.4f\n", best_val_accuracy);
    printf("Final model saved as: retrained_model_best.bin\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <training_data_dir> [base_weights_file] [epochs] [learning_rate]\n", argv[0]);
        printf("Example: %s ../camera/training_data trained_model_weights.bin 50 0.001\n", argv[0]);
        return 1;
    }
    
    const char* data_dir = argv[1];
    const char* base_weights = (argc > 2) ? argv[2] : NULL;
    int epochs = (argc > 3) ? atoi(argv[3]) : 50;
    float learning_rate = (argc > 4) ? atof(argv[4]) : 0.001f;
    
    // Initialize neural network
    NeuralNetworkCUDA nn;
    initialize_network(&nn);
    
    // Load datasets
    Dataset train_dataset = load_camera_dataset(data_dir);
    Dataset val_dataset = load_validation_dataset(data_dir);
    
    if (train_dataset.count == 0) {
        printf("No training data found. Please collect and organize data first.\n");
        free_network(&nn);
        return 1;
    }
    
    // Retrain model
    retrain_model(&nn, &train_dataset, &val_dataset, epochs, learning_rate, base_weights);
    
    // Clean up
    free_dataset(&train_dataset);
    free_dataset(&val_dataset);
    free_network(&nn);
    
    return 0;
}
