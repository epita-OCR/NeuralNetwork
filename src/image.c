#include "perceptron.h"
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

//#define INPUT_SIZE 32*32
#define NUM_CLASSES 25

#include "image.h"


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

// Fonction pour créer un encodage one-hot des labels
void create_one_hot_labels(float** labels_matrix, struct ImageData* dataset, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        int label = dataset[i].label;

        if (labels_matrix[i] == NULL) {
            err(EXIT_FAILURE, "labels_matrix[i] is NULL");
        }
        for (int j = 0; j < NUM_CLASSES; j++) {
            //printf("Trying to acces labels_matrix[%d][%d] || NUM_CLASSES = %i \n", i, j, NUM_CLASSES);
            labels_matrix[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}