#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include "train.h"
#include <err.h>
#include <string.h>

#include "stb_image.h"

#define CHANNELS 1  // 1 pour les images en niveaux de gris, 3 pour RGB
#define MAX_IMAGES 60000
#define NUM_CLASSES 39
#define INPUT_SIZE 28*28
#define BATCH_SIZE 64


// Fonction pour charger une image PNG
float* load_png_image(const char* file_path, int* width, int* height, int* channels) {
    // Charger l'image en niveaux de gris
    unsigned char* data = stbi_load(file_path, width, height, channels, CHANNELS);

    if (data == NULL) {
        printf("Erreur lors du chargement de l'image %s\n", file_path);
        return NULL;
    }

    // Convertir l'image en un tableau de flottants
    float* pixels = (float*) calloc((*width) * (*height), sizeof(float));
    for (int i = 0; i < (*width) * (*height); i++) {
        pixels[i] = data[i] / 255.0f;  // Normalisation des pixels entre 0 et 1
    }

    stbi_image_free(data);  // Libérer la mémoire de l'image chargée
    return pixels;
}

int load_images_from_directory(const char* directory_path, struct ImageData* dataset) {
    DIR* dir = opendir(directory_path);
    if (dir == NULL) {
        err(EXIT_FAILURE,"Erreur lors de l'ouverture du répertoire %s\n", directory_path);
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
                dataset[count].width = width;
                dataset[count].height = height;
                dataset[count].label = 0; // Définir ici la bonne étiquette en fonction du nom de fichier ou du répertoire
                count++;
            }
            // Libérer la mémoire des pixels
            else {
                free(pixels);
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

    struct ImageData *dataset = (struct ImageData *)calloc(MAX_IMAGES, sizeof(struct ImageData));
    if (dataset == NULL) {
        err(1, "Memory allocation failed\n");
        return 1;
    }
    printf("Initializing perceptrons\n");
    init_perceptrons();

    // Charger les images du répertoire "Train". Les images sont dans des sous-dossiers. Il faur donc parcourir les sous-dossiers pour charger les images.
    char* directory_path = "/home/clement/Documents/archive/Train/";
    // Parcourir les sous-dossiers pour charger les images
    char* subdirectories[] = {"&", "@", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "$", "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "#"};

    int dataset_size = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        char* dir[39] = { 0 };
        strcat(dir, directory_path);
        strcat(dir, subdirectories[i]);

        printf("Loading images from directory %s\n", dir);
        dataset_size = load_images_from_directory(dir, dataset);


        printf("Nombre total d'images chargées : %d\n", dataset_size);

        printf("Selecting random images for training\n");
        struct ImageData selected_images[MAX_IMAGES];
        select_random_images(dataset, dataset_size, selected_images, 60000);



        printf("Preparing input arrays and labels for training\n");
        // Préparer les tableaux d'entrée et les labels pour l'entraînement
        float** x_train = calloc(MAX_IMAGES, sizeof(float*));
        float** y_train = calloc(MAX_IMAGES, sizeof(float*));

        printf("Loading and normalizing images for training\n");
        // Charger et normaliser les images pour l'entraînement
        for (int i = 0; i < 60000; i++) {
            if (INPUT_SIZE != selected_images[i].width * selected_images[i].height) {
                err(EXIT_FAILURE, "Image as not the correct size || INPUT_SIZE = %i != size =  %i \n",
                    INPUT_SIZE, selected_images[i].width * selected_images[i].height);
            }
            x_train[i] = selected_images[i].pixels;  // Charger les pixels normalisés
            y_train[i] = calloc(NUM_CLASSES, sizeof(float));  // Allouer les labels en one-hot encoding
            if (y_train[i] == NULL) {
                err(EXIT_FAILURE, "Memory allocation failed for y_train\n");
            }
        }

        // Créer les étiquettes en one-hot encoding
        create_one_hot_labels(y_train, selected_images, MAX_IMAGES);
        printf("Training the model\n");
        train(x_train, y_train, MAX_IMAGES);
        // Libérer la mémoire allouée
        for (int i = 0; i < 60000; i++) {
            free(y_train[i]);
        }
        free(x_train);
        free(y_train);
    }
    int correct = 0;
    while (correct < 900) {
        // Entraîner le modèle


        //Test
        correct = test_model("/home/clement/Documents/archive/Validation/A", 1000);
        printf("Correct predictions: %d\n", correct);
        printf("Saving weights\n");
        save_perceptron_weights(NUM_CLASSES, "weights.txt");
    }



    free(dataset);
    free_percetrons();

    return 0;
}
