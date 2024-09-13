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

#include <stb_image.h>


#define INPUT_SIZE 32*32
#define NUM_CLASSES 36
#define BATCH_SIZE 64
#define MAX_IMAGES 60000


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

// Fonction pour charger une image et la normaliser
int load_image(const char* filepath, float* output_pixels, int target_width, int target_height) {
    int width, height, channels;
    unsigned char* img = stbi_load(filepath, &width, &height, &channels, 1);  // Charger en niveaux de gris

    if (!img) {
        err(0, "Erreur: impossible de charger l'image %s\n", filepath);
        return 0;  // Retourner immédiatement après l'échec du chargement
    }

    if (width != target_width || height != target_height) {
        stbi_image_free(img);  // Libérer la mémoire car l'image n'a pas la bonne taille
        err(0, "Erreur: l'image %s a une taille incorrecte\n", filepath);
        return 0;  // Retourner immédiatement après l'erreur de taille
    }

    // Normaliser les pixels entre -1 et 1
    for (int i = 0; i < INPUT_SIZE; i++) {
        output_pixels[i] = (img[i] - 127.5f) / 127.5f;
    }
    free(img);  // Libérer la mémoire après utilisation
    //stbi_image_free(img);  // Libérer la mémoire après utilisation
    return 1;
}

// Fonction pour charger les images depuis un répertoire et créer des étiquettes
int load_images_from_directory(const char* directory, struct ImageData* dataset, int* dataset_size) {
    DIR* dir = opendir(directory);
    if (!dir) {
        printf("Erreur: impossible d'ouvrir le dossier %s\n", directory);
        return 0;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        char subdirectory[512];
        snprintf(subdirectory, sizeof(subdirectory), "%s/%s", directory, entry->d_name);

        if (is_directory(subdirectory)) {
            // Ignorer les dossiers "." et ".."
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            // Obtenir le label à partir du nom du sous-dossier (par exemple, '0' à '9')
            int label = entry->d_name[0] - '0';

            DIR* subdir = opendir(subdirectory);
            if (!subdir) {
                printf("Erreur: impossible d'ouvrir le sous-dossier %s\n", subdirectory);
                closedir(dir);
                return 0;
            }

            struct dirent* img_entry;
            while ((img_entry = readdir(subdir)) != NULL && *dataset_size < MAX_IMAGES) {
                char filepath[512];
                snprintf(filepath, sizeof(filepath), "%s/%s", subdirectory, img_entry->d_name);

                if (is_regular_file(filepath)) {  // Fichier régulier
                    // Charger l'image et la normaliser
                    if (load_image(filepath, dataset[*dataset_size].pixels, 32, 32) == 1) {
                        dataset[*dataset_size].label = label;
                        (*dataset_size)++;
                    }
                }
            }

            closedir(subdir);
        }
    }

    closedir(dir);
    return 1;
}

// Fonction pour créer un encodage one-hot des labels
void create_one_hot_labels(float** labels_matrix, struct ImageData* dataset, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        int label = dataset[i].label;
        for (int j = 0; j < NUM_CLASSES; j++) {
            labels_matrix[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}

void select_random_images(struct ImageData* dataset, int dataset_size, struct ImageData* selected_images, int num_images) {
    srand(time(NULL));  // Initialiser le générateur de nombres aléatoires
    for (int i = 0; i < num_images; i++) {
        int random_index = rand() % dataset_size;
        selected_images[i] = dataset[random_index];  // Copier l'image sélectionnée aléatoirement
    }
}


struct Perceptron perceptrons[NUM_CLASSES];  // Tableau de 10 perceptrons

// Fonction pour calculer les probabilités softmax
void z_perceptrons(struct Perceptron* perceptrons, float* x, float* ze) {
    float sum_exp = 0.0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        ze[i] = exp(LinarCombi(perceptrons[i], x));
        sum_exp += ze[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        ze[i] /= sum_exp;  // Normalisation pour obtenir les probabilités softmax
    }
}

// Fonction pour calculer la perte de classification avec softmax
float loss_classif_softmax(struct Perceptron* perceptrons, float** x_tab, float** y_tab, size_t n_samples) {
    float total_loss = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        float ze[NUM_CLASSES];
        z_perceptrons(perceptrons, x_tab[i], ze);

        for (int j = 0; j < NUM_CLASSES; j++) {
            total_loss -= y_tab[i][j] * log(ze[j]);
        }
    }
    return total_loss / n_samples;
}

// Fonction de rétropropagation avec softmax pour la classification
void retropopagation_classif_softmax(struct Perceptron* perceptrons, float** x_tab, float** y_tab, size_t n_samples, float eta) {
    for (int i = 0; i < NUM_CLASSES; i++) {
        float* nv_coeffs = (float*) malloc(perceptrons[i].inputs_nb * sizeof(float));
        for (size_t j = 0; j < perceptrons[i].inputs_nb; j++) {
            nv_coeffs[j] = perceptrons[i].coeffs[j];
        }

        float* dy = (float*) malloc(n_samples * sizeof(float));
        float* y_bar = (float*) malloc(n_samples * sizeof(float));

        for (size_t k = 0; k < n_samples; k++) {
            float ze[NUM_CLASSES];
            z_perceptrons(perceptrons, x_tab[k], ze);
            dy[k] = y_tab[k][i] * (1.0 - ze[i]);  // Gradient pour la classe i
        }

        // Mise à jour des coefficients
        for (size_t j = 0; j < perceptrons[i].inputs_nb - 1; j++) {
            float gradient_sum = 0.0;
            for (size_t k = 0; k < n_samples; k++) {
                gradient_sum += dy[k] * x_tab[k][j];
            }
            nv_coeffs[j] += eta * gradient_sum;  // Mise à jour avec la somme des gradients
        }

        // Mise à jour du biais
        float bias_sum = 0.0;
        for (size_t k = 0; k < n_samples; k++) {
            bias_sum += dy[k];
        }
        nv_coeffs[perceptrons[i].inputs_nb - 1] += eta * bias_sum;  // Biais

        // Appliquer les nouveaux coefficients
        for (size_t j = 0; j < perceptrons[i].inputs_nb; j++) {
            perceptrons[i].coeffs[j] = nv_coeffs[j];
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
        for (int j = 0; j < INPUT_SIZE; j++) {
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
        float** x_tab = (float**) malloc(BATCH_SIZE * sizeof(float*));
        float** y_tab = (float**) malloc(BATCH_SIZE * sizeof(float*));
        for (int j = 0; j < BATCH_SIZE; j++) {
            x_tab[j] = (float*) malloc(INPUT_SIZE * sizeof(float));
            y_tab[j] = (float*) malloc(NUM_CLASSES * sizeof(float));
        }

        // Sélectionner un batch aléatoire
        selection_bash(entrainement_imgs, entrainement_chiffres, total_size, x_tab, y_tab, BATCH_SIZE);

        // Effectuer la rétropropagation
        retropopagation_classif_softmax(perceptrons, x_tab, y_tab, BATCH_SIZE, 0.00001);

        // Affichage de la perte toutes les 100 itérations
        if (i % 100 == 0) {
            printf("Loss: %f\n", loss_classif_softmax(perceptrons, entrainement_imgs, entrainement_chiffres, total_size));
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

