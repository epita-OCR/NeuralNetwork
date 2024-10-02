#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "NeuralNetwork.h"

#define MAX_LAYERS 10



float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

void softmax(float* input, int size) {
    float max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

NeuralNetwork* create_neural_network(int num_layers, int* layer_sizes) {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layer_sizes = malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {
        nn->layer_sizes[i] = layer_sizes[i];
    }

    nn->weights = calloc((num_layers - 1), sizeof(float**));
    nn->biases = calloc((num_layers - 1), sizeof(float*));
    nn->activations = calloc(num_layers, sizeof(float*));
    nn->z_values = calloc((num_layers - 1), sizeof(float*));

    for (int i = 0; i < num_layers - 1; i++) {
        nn->weights[i] = calloc(layer_sizes[i + 1], sizeof(float*));
        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            nn->weights[i][j] = calloc(layer_sizes[i], sizeof(float));
            for (int k = 0; k < layer_sizes[i]; k++) {
                nn->weights[i][j][k] = ((float)rand() / RAND_MAX) * 2 - 1;
            }
        }
        nn->biases[i] = calloc(layer_sizes[i + 1], sizeof(float));
        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            nn->biases[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
        nn->z_values[i] = calloc(layer_sizes[i + 1], sizeof(float));
    }

    for (int i = 0; i < num_layers; i++) {
        nn->activations[i] = calloc(layer_sizes[i], sizeof(float));
    }

    return nn;
}

void forward_propagation(NeuralNetwork* nn, float* input) {
    for (int i = 0; i < nn->layer_sizes[0]; i++) {
        nn->activations[0][i] = input[i];
    }

    for (int layer = 1; layer < nn->num_layers; layer++) {
        for (int neuron = 0; neuron < nn->layer_sizes[layer]; neuron++) {
            float sum = nn->biases[layer - 1][neuron];
            for (int prev_neuron = 0; prev_neuron < nn->layer_sizes[layer - 1]; prev_neuron++) {
                sum += nn->weights[layer - 1][neuron][prev_neuron] * nn->activations[layer - 1][prev_neuron];
            }
            nn->z_values[layer - 1][neuron] = sum;
            if (layer == nn->num_layers - 1) {
                nn->activations[layer][neuron] = sum; // For softmax layer
            } else {
                nn->activations[layer][neuron] = sigmoid(sum);
            }
        }
    }

    // Apply softmax to the output layer
    softmax(nn->activations[nn->num_layers - 1], nn->layer_sizes[nn->num_layers - 1]);
}

void backward_propagation(NeuralNetwork* nn, float* target, float learning_rate) {
    int output_layer = nn->num_layers - 1;
    float* output_error = malloc(nn->layer_sizes[output_layer] * sizeof(float));

    // Calculate error for output layer
    for (int i = 0; i < nn->layer_sizes[output_layer]; i++) {
        output_error[i] = nn->activations[output_layer][i] - target[i];
    }

    // Backpropagate the error
    for (int layer = output_layer; layer > 0; layer--) {
        float* current_error = (layer == output_layer) ? output_error : calloc(nn->layer_sizes[layer], sizeof(float));
        float* prev_error = (layer > 1) ? calloc(nn->layer_sizes[layer - 1], sizeof(float)) : NULL;

        for (int i = 0; i < nn->layer_sizes[layer]; i++) {
            float delta = current_error[i];
            if (layer != output_layer) {
                delta *= sigmoid_derivative(nn->z_values[layer - 1][i]);
            }

            // Update weights and biases
            for (int j = 0; j < nn->layer_sizes[layer - 1]; j++) {
                nn->weights[layer - 1][i][j] -= learning_rate * delta * nn->activations[layer - 1][j];
                if (prev_error) {
                    prev_error[j] += delta * nn->weights[layer - 1][i][j];
                }
            }
            nn->biases[layer - 1][i] -= learning_rate * delta;
        }

        if (layer != output_layer)
        {
          //free(current_error);
          }
        if (prev_error) {
            //free(current_error);
            current_error = prev_error;
        }
    }

    //free(output_error);
}

void train(NeuralNetwork* nn, float** inputs, float** targets, int num_samples, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        for (int sample = 0; sample < num_samples; sample++) {
            forward_propagation(nn, inputs[sample]);
            backward_propagation(nn, targets[sample], learning_rate);

            // Calculate loss (cross-entropy)
            for (int i = 0; i < nn->layer_sizes[nn->num_layers - 1]; i++) {
              	//printf("targets[sample][i] = %f\n", targets[sample][i]);
                total_loss -= targets[sample][i] * logf(nn->activations[nn->num_layers - 1][i]);
            }
        }
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / num_samples);
    }
}

int predict(NeuralNetwork* nn, float* input) {
    forward_propagation(nn, input);
    //printf("Prediction: ");
    int predicted_label = 0;
    for (int i = 0; i < nn->layer_sizes[nn->num_layers - 1]; i++) {
        //printf("%f ", nn->activations[nn->num_layers - 1][i]);
        if (nn->activations[nn->num_layers - 1][i] > nn->activations[nn->num_layers - 1][predicted_label]) {
            predicted_label = i;
        }
    }
    //printf("\n");
    return predicted_label;

}

void free_neural_network(NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        for (int j = 0; j < nn->layer_sizes[i + 1]; j++) {
            free(nn->weights[i][j]);
        }
        free(nn->weights[i]);
        free(nn->biases[i]);
        free(nn->z_values[i]);
    }
    for (int i = 0; i < nn->num_layers; i++) {
        free(nn->activations[i]);
    }
    free(nn->weights);
    free(nn->biases);
    free(nn->activations);
    free(nn->z_values);
    free(nn->layer_sizes);
    free(nn);
}

