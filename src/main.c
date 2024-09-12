#include <stdio.h>
#include <stdlib.h>
#include "train.h"

int main() {
    ImageData dataset[MAX_IMAGES];
    int dataset_size = 0;

    // Charger les images du répertoire "Train"
    if (!load_images_from_directory("/home/clement/Documents/test2/archive/Train", dataset, &dataset_size)) {
        printf("Erreur lors du chargement des images.\n");
        return 1;
    }

    printf("Nombre total d'images chargées : %d\n", dataset_size);

    // Sélectionner 60 000 images aléatoires pour l'entraînement
    ImageData selected_images[60000];
    select_random_images(dataset, dataset_size, selected_images, 60000);

    // Préparer les tableaux d'entrée et les labels pour l'entraînement
    float** x_train = (float**) malloc(60000 * sizeof(float*));
    float** y_train = (float**) malloc(60000 * sizeof(float*));

    for (int i = 0; i < 60000; i++) {
        x_train[i] = selected_images[i].pixels;
        y_train[i] = (float*) calloc(NUM_CLASSES, sizeof(float));
    }

    // Créer les étiquettes en one-hot encoding
    create_one_hot_labels(y_train, selected_images, 60000);

    // Entraîner le modèle
    train(x_train, y_train, 60000);

    // Libérer la mémoire allouée
    for (int i = 0; i < 60000; i++) {
        free(y_train[i]);
    }
    free(x_train);
    free(y_train);

    return 0;
}


