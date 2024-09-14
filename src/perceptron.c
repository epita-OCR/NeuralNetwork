#include "perceptron.h"
#include <math.h>  // Pour pow, exp, et log
#include <stdio.h>
#include <stdlib.h>


// Fonction pour initialiser un perceptron avec des coefficients aléatoires
void init_perceptron(struct Perceptron* perceptron, unsigned int inputs_nb) {
    perceptron->inputs_nb = inputs_nb;
    perceptron->coeffs = (float*) malloc(inputs_nb * sizeof(float));
    for (unsigned int i = 0; i < inputs_nb; i++) {
        perceptron->coeffs[i] = (float) rand() / RAND_MAX;  // Coefficients aléatoires entre 0 et 1
    }
}


// Fonction pour calculer la combinaison linéaire
float LinarCombi(struct Perceptron perceptron, float* x) {
    float arg = perceptron.coeffs[perceptron.inputs_nb - 1];  // Dernier coefficient (biais)
    for (unsigned int i = 0; i < perceptron.inputs_nb - 1; i++) {
        arg += perceptron.coeffs[i] * x[i];  // Combinaison linéaire des coefficients et des entrées
        if (x[i] != 0) {
            //printf("x[%d] = %f\narg = %f", i, x[i], arg);
        }
    }
    return arg;
}

// Fonction de calcul de la perte pour la régression (Mean Squared Error)
float loss_regression(struct Perceptron perceptron, float** features, size_t features_len, float* target) {
    float total_loss = 0.0;
    for (size_t i = 0; i < features_len; i++) {
        float predicted_value = LinarCombi(perceptron, features[i]);
        float error = predicted_value - target[i];
        total_loss += pow(error, 2);  // Erreur quadratique
    }
    return total_loss / features_len;  // Moyenne des erreurs quadratiques
}

// Fonction de calcul de la perte pour la classification (Cross-Entropy Loss)
float loss_classification(struct Perceptron perceptron, float** features, size_t features_len, float* target) {
    float total_loss = 0.0;
    for (size_t i = 0; i < features_len; i++) {
        float predicted_value = LinarCombi(perceptron, features[i]);
        float y_bar = 1.0 / (1.0 + exp(-predicted_value));  // Fonction sigmoïde
        float log_y_bar = log(y_bar);
        total_loss -= target[i] * log_y_bar;  // Calcul de la perte d'entropie croisée
    }
    return total_loss / features_len;
}

// Fonction de rétropropagation pour la régression (gradient descent)
void retropropagation_regression(struct Perceptron* perceptron, float** features, float* target, size_t features_len, float eta) {
    size_t d = perceptron->inputs_nb - 1;
    float* new_coeffs = (float*) malloc((perceptron->inputs_nb) * sizeof(float));

    // Copie des coefficients actuels
    for (size_t i = 0; i < perceptron->inputs_nb; i++) {
        new_coeffs[i] = perceptron->coeffs[i];
    }

    // Calcul des gradients et mise à jour des coefficients
    for (size_t i = 0; i < d; i++) {
        float gradient_sum = 0.0;
        for (size_t j = 0; j < features_len; j++) {
            float predicted_value = LinarCombi(*perceptron, features[j]);
            float error = 2.0 * (predicted_value - target[j]);
            gradient_sum += error * features[j][i];
        }
        new_coeffs[i] -= eta * gradient_sum;
    }

    // Mise à jour du biais
    float gradient_bias = 0.0;
    for (size_t j = 0; j < features_len; j++) {
        float predicted_value = LinarCombi(*perceptron, features[j]);
        gradient_bias += 2.0 * (predicted_value - target[j]);
    }
    new_coeffs[perceptron->inputs_nb - 1] -= eta * gradient_bias;

    // Mise à jour des coefficients
    for (size_t i = 0; i < perceptron->inputs_nb; i++) {
        perceptron->coeffs[i] = new_coeffs[i];
    }

    free(new_coeffs);
}

// Fonction de rétropropagation pour la classification (softmax)
void retropropagation_classification(struct Perceptron* perceptron, float** features, size_t features_len, float* target, float eta) {
    float* new_coeffs = (float*) malloc(perceptron->inputs_nb * sizeof(float));

    for (size_t i = 0; i < perceptron->inputs_nb; i++) {
        new_coeffs[i] = perceptron->coeffs[i];
    }

    for (size_t i = 0; i < perceptron->inputs_nb; i++) {
        float gradient_sum = 0.0;
        for (size_t j = 0; j < features_len; j++) {
            float predicted_value = LinarCombi(*perceptron, features[j]);
            float y_bar = 1.0 / (1.0 + exp(-predicted_value));
            float error = target[j] - y_bar;
            gradient_sum += error * features[j][i];
        }
        new_coeffs[i] += eta * gradient_sum;
    }

    for (size_t i = 0; i < perceptron->inputs_nb; i++) {
        perceptron->coeffs[i] = new_coeffs[i];
    }

    free(new_coeffs);
}

