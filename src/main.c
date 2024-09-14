#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include "train.h"
#include <err.h>
#include "stb_image.h"

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define CHANNELS 1  // 1 pour les images en niveaux de gris, 3 pour RGB
#define MAX_IMAGES 60000
#define NUM_CLASSES 36

// Fonction pour charger une image PNG
float* load_png_image(const char* file_path, int* width, int* height, int* channels) {
    // Charger l'image en niveaux de gris
    unsigned char* data = stbi_load(file_path, width, height, channels, CHANNELS);

    if (data == NULL) {
        printf("Erreur lors du chargement de l'image %s\n", file_path);
        return NULL;
    }

    // Convertir l'image en un tableau de flottants
    float* pixels = (float*) malloc((*width) * (*height) * sizeof(float));
    for (int i = 0; i < (*width) * (*height); i++) {
        pixels[i] = data[i] / 255.0f;  // Normalisation des pixels entre 0 et 1
    }

    stbi_image_free(data);  // Libérer la mémoire de l'image chargée
    return pixels;
}

int load_images_from_directory(const char* directory_path, struct ImageData* dataset) {
    DIR* dir = opendir(directory_path);
    if (dir == NULL) {
        printf("Erreur lors de l'ouverture du répertoire %s\n", directory_path);
        return 0;
    }

    struct dirent* entry;
    int count = 0;

    while ((entry = readdir(dir)) != NULL) {
        // Ignorer les répertoires spéciaux "." et ".."
        if (entry->d_type == DT_REG) {
            char filepath[256];
            snprintf(filepath, sizeof(filepath), "%s/%s", directory_path, entry->d_name);

            // Charger chaque image en appelant `load_png_image`
            int width, height, channels;
            float* pixels = load_png_image(filepath, &width, &height, &channels);

            if (pixels != NULL && count < MAX_IMAGES) {
                dataset[count].pixels = pixels;
                dataset[count].label = 0; // Définir ici la bonne étiquette en fonction du nom de fichier ou du répertoire
                count++;
            }
        }
    }

    closedir(dir);
    //dataset_size = count;
    return count;
}

int main() {
    printf("Hey\n");
    printf("Allocating %lu bytes for dataset\n", MAX_IMAGES * sizeof(struct ImageData));

    struct ImageData *dataset = (struct ImageData *)malloc(MAX_IMAGES * sizeof(struct ImageData));
    if (dataset == NULL) {
        err(1, "Memory allocation failed\n");
        return 1;
    }
    printf("Initializing perceptrons\n");
    init_perceptrons();

    // Charger les images du répertoire "Train". Les images sont dans des sous-dossiers. Il faur donc parcourir les sous-dossiers pour charger les images.
    char* directory_path = "/home/clement/Documents/test2/archive/Train/A";

    printf("Loading images from directory %s\n", directory_path);
    int dataset_size = load_images_from_directory(directory_path, dataset);


    printf("Nombre total d'images chargées : %d\n", dataset_size);

    printf("Selecting random images for training\n");
    struct ImageData selected_images[MAX_IMAGES];
    select_random_images(dataset, dataset_size, selected_images, 60000);

    printf("Preparing input arrays and labels for training\n");
    // Préparer les tableaux d'entrée et les labels pour l'entraînement
    float** x_train = (float**) malloc(MAX_IMAGES * sizeof(float*));
    float** y_train = (float**) malloc(MAX_IMAGES * sizeof(float*));

    printf("Loading and normalizing images for training\n");
    // Charger et normaliser les images pour l'entraînement
    for (int i = 0; i < 60000; i++) {
        x_train[i] = selected_images[i].pixels;  // Charger les pixels normalisés
        y_train[i] = (float*) calloc(NUM_CLASSES, sizeof(float));  // Allouer les labels en one-hot encoding
    }

    // Créer les étiquettes en one-hot encoding
    create_one_hot_labels(y_train, selected_images, 60000);
    printf("Training the model\n");
    // Entraîner le modèle
    train(x_train, y_train, 60000);

    // Libérer la mémoire allouée
    for (int i = 0; i < 60000; i++) {
        free(y_train[i]);
    }
    free(x_train);
    free(y_train);
    free(dataset);

    return 0;
}
