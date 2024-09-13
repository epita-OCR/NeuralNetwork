#include "perceptron_tests.h"
#include <criterion/criterion.h>
#include "../src/perceptron.h"
#include <stdio.h>

// Function prototypes
float LinarCombi(struct Perceptron perceptron, float* x);

Test(LinarCombi, CalculatesCorrectly) {
    float coeffs[] = {1.0, 2.0, 3.0}; // Coefficients including bias
    struct Perceptron perceptron = {coeffs, 3};
    float x[] = {4.0, 5.0}; // Input features

    float result = LinarCombi(perceptron, x);
    cr_assert_float_eq(result, 1.0 + 2.0 * 4.0 + 3.0 * 5.0, 1e-6);
}

Test(LinarCombi, HandlesZeroCoefficients) {
    float coeffs[] = {0.0, 0.0, 0.0}; // Coefficients including bias
    struct Perceptron perceptron = {coeffs, 3};
    float x[] = {4.0, 5.0}; // Input features

    float result = LinarCombi(perceptron, x);
    cr_assert_float_eq(result, 0.0, 1e-6);
}

Test(LinarCombi, HandlesNegativeCoefficients) {
    float coeffs[] = {-1.0, -2.0, -3.0}; // Coefficients including bias
    struct Perceptron perceptron = {coeffs, 3};
    float x[] = {4.0, 5.0}; // Input features

    float result = LinarCombi(perceptron, x);
    cr_assert_float_eq(result, -1.0 + (-2.0) * 4.0 + (-3.0) * 5.0, 1e-6);
}

Test(LinarCombi, HandlesEmptyInput) {
    float coeffs[] = {1.0}; // Only bias
    struct Perceptron perceptron = {coeffs, 1};
    float x[] = {}; // No input features

    float result = LinarCombi(perceptron, x);
    cr_assert_float_eq(result, 1.0, 1e-6);
}