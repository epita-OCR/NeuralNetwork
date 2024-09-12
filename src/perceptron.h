#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stddef.h>  // Pour size_t

// Structure d'un perceptron
struct Perceptron {
    float* coeffs;    // Tableau des coefficients du perceptron (y compris le biais)
    size_t inputs_nb; // Nombre d'entrées (y compris le biais)
};

/**
 * Calcule la combinaison linéaire des entrées avec les coefficients du perceptron.
 *
 * @param perceptron La structure du perceptron
 * @param x Tableau des entrées (features)
 * @return La combinaison linéaire des coefficients et des entrées
 */
float LinarCombi(struct Perceptron perceptron, float* x);

/**
 * Calcule la perte de régression (Mean Squared Error).
 *
 * @param perceptron Le perceptron utilisé
 * @param features Matrice des caractéristiques d'entrée
 * @param features_len Nombre d'exemples de données
 * @param target Tableau des valeurs cibles
 * @return La perte MSE
 */
float loss_regression(struct Perceptron perceptron, float** features, size_t features_len, float* target);

/**
 * Calcule la perte d'entropie croisée (Cross-Entropy Loss) pour la classification.
 *
 * @param perceptron Le perceptron utilisé
 * @param features Matrice des caractéristiques d'entrée
 * @param features_len Nombre d'exemples de données
 * @param target Tableau des valeurs cibles (labels)
 * @return La perte d'entropie croisée
 */
float loss_classification(struct Perceptron perceptron, float** features, size_t features_len, float* target);

/**
 * Effectue la rétropropagation pour la régression en mettant à jour les coefficients.
 *
 * @param perceptron Le perceptron à mettre à jour
 * @param features Matrice des caractéristiques d'entrée
 * @param target Tableau des valeurs cibles
 * @param features_len Nombre d'exemples de données
 * @param eta Le taux d'apprentissage (learning rate)
 */
void retropropagation_regression(struct Perceptron* perceptron, float** features, float* target, size_t features_len, float eta);

/**
 * Effectue la rétropropagation pour la classification en mettant à jour les coefficients.
 *
 * @param perceptron Le perceptron à mettre à jour
 * @param features Matrice des caractéristiques d'entrée
 * @param target Tableau des valeurs cibles (labels)
 * @param features_len Nombre d'exemples de données
 * @param eta Le taux d'apprentissage (learning rate)
 */
void retropropagation_classification(struct Perceptron* perceptron, float** features, size_t features_len, float* target, float eta);

#endif // PERCEPTRON_H

