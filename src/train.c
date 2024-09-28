#include "perceptron.h"
#include "train.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <dirent.h>      // Pour opendir, readdir, closedir
#include <sys/types.h>   // Pour les types de fichiers
#include <sys/stat.h>    // Pour DT_DIR et DT_REG
                         //
#define STB_IMAGE_IMPLEMENTATION
#include <err.h>

#include "stb_image.h"

#define INPUT_SIZE 28*28




int is_directory(const char* path) {
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) {
        return 0;
    }
    return S_ISDIR(statbuf.st_mode);
}

// Fonction pour vérifier si un chemin est un fichier régulier
int is_regular_file(const char* path) {
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) {
        return 0;
    }
    return S_ISREG(statbuf.st_mode);
}



// Fonction pour créer un encodage one-hot des labels
void create_one_hot_labels(float** labels_matrix, struct ImageData* dataset, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        int label = dataset[i].label;

        if (labels_matrix[i] == NULL) {
            err(EXIT_FAILURE, "labels_matrix[i] is NULL");
        }
        for (int j = 0; j < NUM_CLASSES; j++) {
            //printf("Trying to acces labels_matrix[%d][%d] \n", i, j);
            labels_matrix[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}

void select_random_images(struct ImageData* dataset, int dataset_size, struct ImageData* selected_images, int num_images) {
    if (dataset_size == 0) {
        err(EXIT_FAILURE, "Dataset is empty - avoid division by zero");
    }
    srand(time(NULL));  // Initialiser le générateur de nombres aléatoires
    for (int i = 0; i < num_images; i++) {
        int random_index = rand() % dataset_size;
        selected_images[i] = dataset[random_index];  // Copier l'image sélectionnée aléatoirement
    }
}


struct Perceptron perceptrons[NUM_CLASSES]; // Tableau de 10 perceptrons

// initialisation des perceptrons
void init_perceptrons() {
    for (int i = 0; i < NUM_CLASSES; i++) {
        init_perceptron(&perceptrons[i], INPUT_SIZE + 1);  // Initialisation avec un biais
    }
}

// Fonction pour calculer les probabilités softmax
void z_perceptrons(struct Perceptron* perceptrons, float* x, double* ze) {
    long double sum_exp = 0.0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (isnan(perceptrons[i].coeffs[0])) {
            err(EXIT_FAILURE, "perceptrons[i].coeffs is NULL");
        }
        float predicted_value = LinarCombi(perceptrons[i], x); // Calcul de la combinaison linéaire
        //printf("predicted_value = %f || x = %d \n", predicted_value, i);
        // Verifie si la valeur prédite est un nombre
        if (isnan(predicted_value)) {
            err(EXIT_FAILURE,"predicted_value is nan classes = %i || with x = %f , ze = %f", i, x, ze);
        }
        ze[i] = exp(predicted_value); // Calcul de la combinaison linéaire
        sum_exp += ze[i]; // Somme des exponentielles
        //printf("sum_exp = %f #42", sum_exp, i, ze[i]);
        if( sum_exp == INFINITY || sum_exp == -INFINITY) {
            err(EXIT_FAILURE, "sum_exp is inf || ze[%d] = %f || predicted_value = %f", i, ze[i], predicted_value);
        }
        if (isnan(sum_exp)) {
            err(EXIT_FAILURE, "sum_exp is nan");
        }
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        ze[i] /= sum_exp;  // Normalisation pour obtenir les probabilités softmax
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        if(isnan(ze[i])) {
            err(EXIT_FAILURE, "ze is nan for classes = %i || sum_exp = %f", i, sum_exp);
        }
    }
}

// Fonction pour calculer la perte de classification avec softmax
float loss_classif_softmax(struct Perceptron* perceptrons, float** x_tab, float** y_tab, size_t n_samples) {
    float total_loss = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        double ze[NUM_CLASSES];
        z_perceptrons(perceptrons, x_tab[i], ze); // Calcul des probabilités softmax

        for (int j = 0; j < NUM_CLASSES; j++) {
            total_loss -= y_tab[i][j] * log(ze[j]); // Calcul de la perte d'entropie croisée
        }
    }
    return total_loss / n_samples;
}

// Fonction de rétropropagation avec softmax pour la classification
void retropopagation_classif_softmax(struct Perceptron* perceptrons, float** x_tab, float** y_tab, size_t n_samples, float eta) {
    if (isnan(perceptrons[0].coeffs[0])) {
        err(EXIT_FAILURE, "perceptrons[0].coeffs is NULL (retropopagation_classif_softmax 1)");
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        float* nv_coeffs = (float*) calloc(perceptrons[i].inputs_nb, sizeof(float));
        for (size_t j = 0; j < perceptrons[i].inputs_nb; j++) {
            nv_coeffs[j] = perceptrons[i].coeffs[j];
        }
        if (isnan(nv_coeffs[0])) {
            err(EXIT_FAILURE, "nv_coeffs is NULL (retropopagation_classif_softmax 2)");
        }

        float* dy = calloc(n_samples, sizeof(float)); // Gradient pour chaque échantillon
        float* y_bar = calloc(n_samples, sizeof(float)); // Prédictions

        for (size_t k = 0; k < n_samples; k++) {
            double ze[NUM_CLASSES]; // Probabilités softmax
            z_perceptrons(perceptrons, x_tab[k], ze); // Calcul des probabilités softmax
            dy[k] = y_tab[k][i] * (1.0 - ze[i]);  // Gradient pour la classe i
            if (isnan(ze[i])) {
                err(EXIT_FAILURE, "ze is NULL (retropopagation_classif_softmax) || ze[%d] = %f", i, ze[i]);
            }
        }

        if (isnan(dy[0])) {
            err(EXIT_FAILURE, "dy is NULL (retropopagation_classif_softmax)");
        }

        // Mise à jour des coefficients
        for (size_t j = 0; j < perceptrons[i].inputs_nb - 1; j++) {
            float gradient_sum = 0.0;
            for (size_t k = 0; k < n_samples; k++) {
                gradient_sum += dy[k] * x_tab[k][j]; // Calcul de la somme des gradients
            }
            nv_coeffs[j] += eta * gradient_sum;  // Mise à jour avec la somme des gradients
        }

        if (isnan(nv_coeffs[0])) {
            err(EXIT_FAILURE, "nv_coeffs is NULL (retropopagation_classif_softmax 3)");
        }

        // Mise à jour du biais
        float bias_sum = 0.0;
        for (size_t k = 0; k < n_samples; k++) {
            bias_sum += dy[k]; // Somme des gradients
        }
        nv_coeffs[perceptrons[i].inputs_nb - 1] += eta * bias_sum;  // Biais

        if(isnan(nv_coeffs[0])) {
            err(EXIT_FAILURE, "nv_coeffs is NULL (retropopagation_classif_softmax 4)");
        }

        // Appliquer les nouveaux coefficients
        for (size_t j = 0; j < perceptrons[i].inputs_nb; j++) {
            perceptrons[i].coeffs[j] = nv_coeffs[j];
        }
        if (isnan(perceptrons[i].coeffs[0])) {
            err(EXIT_FAILURE, "perceptrons[i].coeffs is NULL (retropopagation_classif_softmax)");
        }

        // Libération de la mémoire
        free(nv_coeffs);
        free(dy);
        free(y_bar);
    }
}

// Fonction de sélection aléatoire d'un batch
void selection_bash(float** x_tot, float** y_tot, size_t total_size, float** x_tab, float** y_tab, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        int index = rand() % total_size;
        if (x_tab[i] == NULL) {
            err(EXIT_FAILURE, "x_tab[i] is NULL");
        }
        if (y_tab[i] == NULL) {
            err(EXIT_FAILURE, "y_tab[i] is NULL");
        }
        for (int j = 0; j < INPUT_SIZE; j++) {
            //printf("Trying to acces x_tot[%d][%d] || x_tot = %f || INPUT_SIZE = %i \n", index, j, x_tot[index][j], INPUT_SIZE);
            x_tab[i][j] = x_tot[index][j];  // Copie des entrées
        }
        for (int j = 0; j < NUM_CLASSES; j++) {
            y_tab[i][j] = y_tot[index][j];  // Copie des labels
        }
    }
}

// Fonction principale d'entraînement
void train(float** entrainement_imgs, float** entrainement_chiffres, size_t total_size) {
    for (int i = 0; i < 1000; i++) {
        float** x_tab = (float**) calloc(BATCH_SIZE, sizeof(float*)); // Tableau des entrées
        float** y_tab = (float**) calloc(BATCH_SIZE, sizeof(float*)); // Tableau des labels
        for (int j = 0; j < BATCH_SIZE; j++) {
            x_tab[j] = (float*) calloc(INPUT_SIZE, sizeof(float)); //  Tableau des entrées
            y_tab[j] = (float*) calloc(NUM_CLASSES, sizeof(float));
        }

        // Sélectionner un batch aléatoire
        selection_bash(entrainement_imgs, entrainement_chiffres, total_size, x_tab, y_tab, BATCH_SIZE);

        // Effectuer la rétropropagation
        //printf(" i = %d || BATCH_SIZE = %d || eta = %f \n", i, BATCH_SIZE, 0.00001);
        retropopagation_classif_softmax(perceptrons, x_tab, y_tab, BATCH_SIZE, 0.00001);

        // Affichage de la perte toutes les 100 itérations
        if (i % 100 == 0) {
            printf("i : %i || Loss: %f\n", i, loss_classif_softmax(perceptrons, entrainement_imgs, entrainement_chiffres, total_size));
        }

        // Libérer la mémoire allouée pour le batch
        for (int j = 0; j < BATCH_SIZE; j++) {
            free(x_tab[j]);
            free(y_tab[j]);
        }
        free(x_tab);
        free(y_tab);
    }
}

int test_model(const char* directory_path, int num_images) {
    struct ImageData* test_dataset = calloc(num_images, sizeof(struct ImageData));
    if (test_dataset == NULL) {
        err(EXIT_FAILURE, "Memory allocation failed for test dataset\n");
    }

    int test_dataset_size = load_images_from_directory(directory_path, test_dataset);
    if (test_dataset_size < num_images) {
        num_images = test_dataset_size;
    }

    int correct_predictions = 0;

    for (int i = 0; i < num_images; i++) {

        double* ze = calloc(NUM_CLASSES, sizeof(double));
        z_perceptrons(perceptrons, test_dataset[i].pixels, ze);

        // Trouver l'indice de la classe prédite
        int predicted_label = 0;
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (ze[j] > ze[predicted_label]) {
                predicted_label = j;
            }
        }
        free(ze);

        // Comparer avec l'étiquette réelle
        if (predicted_label == test_dataset[i].label) {
            correct_predictions++;
        }

    }

    free(test_dataset);
    return correct_predictions;
}

void free_percetrons() {
    for (int i = 0; i < NUM_CLASSES; i++) {
        free(perceptrons[i].coeffs);
    }
}


void save_perceptron_weights(int num_perceptrons, const char* file_path) {
    FILE* file = fopen(file_path, "w");  // Ouvrir le fichier en mode écriture
    if (file == NULL) {
        perror("Failed to open file for writing");
        return;
    }

    // Sauvegarder les poids de chaque perceptron
    for (int i = 0; i < num_perceptrons; i++) {
        fprintf(file, "Perceptron %d:\n", i);
        for (int j = 0; j < perceptrons[i].inputs_nb; j++) {
            fprintf(file, "%f ", perceptrons[i].coeffs[j]);
        }
        fprintf(file, "\n");  // Saut de ligne après chaque perceptron
    }

    fclose(file);  // Fermer le fichier
    printf("Weights saved successfully to %s\n", file_path);
}
