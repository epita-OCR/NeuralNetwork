//
// Created by clement on 01/10/24.
//

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

typedef struct {
    int num_layers;
    int* layer_sizes;
    float*** weights;
    float** biases;
    float** activations;
    float** z_values;
} NeuralNetwork;

float sigmoid(float x);

float sigmoid_derivative(float x);

void softmax(float* input, int size);

NeuralNetwork* create_neural_network(int num_layers, int* layer_sizes);

void forward_propagation(NeuralNetwork* nn, float* input);

void backward_propagation(NeuralNetwork* nn, float* target, float learning_rate);

void train(NeuralNetwork* nn, float** inputs, float** targets, int num_samples, int epochs, float learning_rate);

int predict(NeuralNetwork* nn, float* input);

void free_neural_network(NeuralNetwork* nn);

void shuffle_dataset(float** inputs, float** targets, int num_samples);



#endif //NEURALNETWORK_H
