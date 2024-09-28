#include "perceptron.h"

#include <err.h>
#include <math.h>  // Pour pow, exp, et log
#include <stdio.h>
#include <stdlib.h>


// Fonction pour initialiser un perceptron avec des coefficients aléatoires
void init_perceptron(struct Perceptron* perceptron, unsigned int inputs_nb) {
    perceptron->inputs_nb = inputs_nb;
    perceptron->coeffs = (float*) calloc(inputs_nb, sizeof(float));
    for (unsigned int i = 0; i < inputs_nb; i++) {
        perceptron->coeffs[i] = (float) rand() / RAND_MAX;  // Coefficients aléatoires entre 0 et 1
    }
    if (isnan(perceptron->coeffs[0]) || perceptron->coeffs == NULL) {
        err(EXIT_FAILURE, "perceptron->coeffs is NULL");
    }
}


// Fonction pour calculer la combinaison linéaire
float LinarCombi(struct Perceptron perceptron, float* x) {
    float arg = perceptron.coeffs[perceptron.inputs_nb - 1];  // Dernier coefficient (biais)
    for (unsigned int i = 0; i < perceptron.inputs_nb - 1; i++) {
        arg += perceptron.coeffs[i] * x[i];  // Combinaison linéaire des coefficients et des entrées
        if (perceptron.coeffs[i] == INFINITY || perceptron.coeffs[i] == -INFINITY || x[i] == INFINITY || x[i] == -INFINITY) {
            err(EXIT_FAILURE, "LinarCombi is nan beacause of Infinit value");
        }
        else if (isnan(perceptron.coeffs[i]) || x == NULL) {
            err(EXIT_FAILURE, "LinarCombi is nan beacause of NULL value");
        }
        if (isnan(arg)) {
            err(EXIT_FAILURE, "LinarCombi is nan beacause of arg value");
        }
    }
    return arg;
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

// Fonction de rétropropagation pour la classification (softmax)
void retropropagation_classification(struct Perceptron* perceptron, float** features, size_t features_len, float* target, float eta) {
    float* new_coeffs = (float*) calloc(perceptron->inputs_nb, sizeof(float));

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

