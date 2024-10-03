#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
//#include "train.h"
#include <err.h>
#include <string.h>
#include <time.h>
#include "image.h"
#include <sys/wait.h>  // For wait
#include <unistd.h>    // For fork

#include "NeuralNetwork.h"


#include "stb_image.h"

#define CHANNELS 1  // 1 pour les images en niveaux de gris, 3 pour RGB
#define MAX_IMAGES 60000
#define NUM_CLASSES 26
#define INPUT_SIZE 32*32
#define BATCH_SIZE 64


float* resize_image_nearest_neighbor(float* input_pixels, int old_width, int old_height, int new_width, int new_height) {
    float* output_pixels = (float*) calloc(new_width * new_height, sizeof(float));
    if (output_pixels == NULL) {
        printf("Erreur lors de l'allocation de mémoire pour l'image redimensionnée\n");
        return NULL;
    }

    for (int y = 0; y < new_height; y++) {
        int nearest_y = y * old_height / new_height;
        for (int x = 0; x < new_width; x++) {
            int nearest_x = x * old_width / new_width;
            output_pixels[y * new_width + x] = input_pixels[nearest_y * old_width + nearest_x];
        }
    }

    return output_pixels;
}


// Fonction pour charger une image PNG
float* load_png_image(const char* file_path, int* width, int* height, int* channels) {
    // Charger l'image en niveaux de gris
    unsigned char* data = stbi_load(file_path, width, height, channels, CHANNELS);

    if (data == NULL) {
        printf("Erreur lors du chargement de l'image %s\n", file_path);
        return NULL;
    }

    int old_width = *width;
    int old_height = *height;

    // Convertir l'image en un tableau de flottants
    float* pixels = (float*) calloc(old_width * old_height, sizeof(float));
    for (int i = 0; i < old_width * old_height; i++) {
        pixels[i] = data[i] / 255.0f;  // Normalisation des pixels entre 0 et 1
    }

    stbi_image_free(data);  // Libérer la mémoire de l'image chargée

    // Redimensionner l'image à 32x32 pixels
    int new_width = 32;
    int new_height = 32;
    float* resized_pixels = resize_image_nearest_neighbor(pixels, old_width, old_height, new_width, new_height);

    free(pixels);  // Libérer la mémoire de l'image originale

    if (resized_pixels == NULL) {
        printf("Erreur lors du redimensionnement de l'image %s\n", file_path);
        return NULL;
    }

    *width = new_width;
    *height = new_height;

    return resized_pixels;
}



void get_filename_without_extension(const char *filepath, char *filename)
{
    const char *slash = strrchr(filepath, '/');      // Pour Unix/Linux
    const char *backslash = strrchr(filepath, '\\'); // Pour Windows

    // Choisir le bon séparateur de chemin
    const char *basename = slash ? slash + 1 : filepath;
    if (backslash && backslash > basename)
    {
        basename = backslash + 1;
    }

    // Trouver le dernier point
    const char *dot = strrchr(basename, '.');

    // Copier le nom du fichier sans l'extension
    if (dot)
    {
        strncpy(filename, basename, dot - basename);
        filename[dot - basename] = '\0'; // Ajouter le caractère de fin de chaîne
    }
    else
    {
        strcpy(filename, basename); // Aucun point trouvé, copier le nom entier
    }
}

int load_images_from_directory(const char* directory_path, struct ImageData* dataset, size_t actual_size) {
    DIR* dir = opendir(directory_path);
    if (dir == NULL) {
        err(EXIT_FAILURE,"Erreur lors de l'ouverture du répertoire %s\n", directory_path);
    }

    struct dirent* entry;
    int count = actual_size;
    int res = 0;
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

                char *dir[1];
                get_filename_without_extension(directory_path, dir);
                dataset[count].label = ((char) dir[0]) - 'A'; // Définir ici la bonne étiquette en fonction du nom de fichier ou du répertoire
                //printf("FilePath : %s || Domc name = %s || n = %i\n", filepath, dir, ((char) dir[0]) - 97);
                //dataset[count].filename = filepath;
                if (INPUT_SIZE != dataset[count].width * dataset[count].height) {
                    err(EXIT_FAILURE, "Image as not the correct size || INPUT_SIZE = %i != size =  %i || dir = %s\n",
                        INPUT_SIZE, dataset[count].width * dataset[count].height, dir);
                }
                count++;
                res++;
            }
            // Libérer la mémoire des pixels
            else {
                free(pixels);
            }
        }
    }
    closedir(dir);
    //dataset_size = count;
    //printf("res = %i\n", res);
    return res;
}


void start_train(NeuralNetwork *nn) {
    printf("Hey\n");
    printf("Allocating %lu bytes for dataset\n", MAX_IMAGES * sizeof(struct ImageData));

    printf("Initializing perceptrons\n");

    // Charger les images du répertoire "Train". Les images sont dans des sous-dossiers. Il faur donc parcourir les sous-dossiers pour charger les images.
    char* directory_path = "/home/clement.forget/NeuralNetwork/data/training_data/";
    // Parcourir les sous-dossiers pour charger les images
    char* subdirectories[] = {"A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};

    int dataset_size = 0;
    //pid_t pids[NUM_CLASSES];
    struct ImageData *dataset = calloc(MAX_IMAGES, sizeof(struct ImageData));
    for (int i = 0; i < NUM_CLASSES; i++) {
        //if ((pids[i] = fork()) == 0) {
        //Child process
        char dir[256];
        snprintf(dir, sizeof(dir), "%s%s", directory_path, subdirectories[i]);


        if (dataset == NULL) {
            err(1, "Memory allocation failed\n");
        }

        printf("Loading images from directory %s\n", dir);
        printf("dataset_size = %i\n", dataset_size);
        if (dataset_size >= MAX_IMAGES) {
            err(EXIT_FAILURE, "Maximum number of images reached\n");
        }
        dataset_size += load_images_from_directory(dir, dataset, dataset_size);
    }
            printf("Nombre total d'images chargées : %d\n", dataset_size);



            //printf("Selecting random images for training\n");
            //struct ImageData selected_images[dataset_size];
            //select_random_images(dataset, dataset_size, selected_images, dataset_size);

            printf("Preparing input arrays and labels for training\n");
            // Préparer les tableaux d'entrée et les labels pour l'entraînement
            float** x_train = calloc(dataset_size, sizeof(float*));
            float** y_train = calloc(dataset_size, sizeof(float*));

            printf("Loading and normalizing images for training\n");
            // Charger et normaliser les images pour l'entraînement
            for (int i = 0; i < dataset_size; i++) {
                if (INPUT_SIZE != dataset[i].width * dataset[i].height && dataset[i].pixels != NULL) {
                    err(EXIT_FAILURE, "Image as not the correct size || INPUT_SIZE = %i != size =  %i\n",
                        INPUT_SIZE, dataset[i].width * dataset[i].height);
                }
                x_train[i] = dataset[i].pixels;  // Charger les pixels normalisés
                y_train[i] = calloc(NUM_CLASSES, sizeof(float));  // Allouer les labels en one-hot encoding
                if (y_train[i] == NULL) {
                    err(EXIT_FAILURE, "Memory allocation failed for y_train\n");
                }
            }

            printf("Creating one-hot labels for training\n");
            create_one_hot_labels(y_train, dataset, dataset_size);
            printf("Training the model\n");
            // Shuffle the dataset
            shuffle_dataset(x_train, y_train, dataset_size);
            train(nn, x_train, y_train, dataset_size, 100, 2);

            // Libérer la mémoire allouée
            for (int i = 0; i < dataset_size; i++) {
                free(y_train[i]);
            }
            free(x_train);
            free(y_train);
            free(dataset);



    }


//Test the model
int test_model(NeuralNetwork *nn, const char* directory_path, int num_images) {

    // Charger les images du répertoire "Validation". Les images sont dans des sous-dossiers. Il faur donc parcourir les sous-dossiers pour charger les images.
    //char* directory_path = "/home/clement/Documents/archive/Validation/";
    // Parcourir les sous-dossiers pour charger les images
   char* subdirectories[] = {"A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"}; 
    float total_correct = 0;
    float total_size = 0;
    for (int letter = 0; letter < NUM_CLASSES; letter++) {
        float correct = 0;
        char* dir[39] = { 0 };
        strcat(dir, directory_path);
        strcat(dir, subdirectories[letter]);

        int dataset_size = 0;

        struct ImageData *dataset = calloc(MAX_IMAGES, sizeof(struct ImageData));
        if (dataset == NULL) {
            err(1, "Memory allocation failed\n");

        }

        printf("Loading images from directory %s\n", dir);
        dataset_size = load_images_from_directory(dir, dataset, 0);
        printf("dataset_size = %i\n", dataset_size);
        total_size += dataset_size;

        //printf("Nombre total d'images chargées : %d\n", dataset_size);
        // Use predict function to predict the label of the image
        for (int i = 0; i < dataset_size; i++) {
            float* input = dataset[i].pixels;
            if (dataset[i].pixels == NULL) {
               err(EXIT_FAILURE, "Test -- Image as not the correct size || INPUT_SIZE = %i != size =  %i || dir = %s\n",
                    INPUT_SIZE, dataset[i].width * dataset[i].height, dir);
            }
            int predicted_label = predict(nn, input);
            printf("For label %c, predicted label = %c\n", 'A' + dataset[i].label, 'A' + predicted_label);
            if (predicted_label == dataset[i].label) {
                correct++;
            }
        }
        total_correct += correct;
        printf("Correct predictions for letter %c: %f\n", 'A' + letter, correct);
    }
    printf("Number total of images : %f\n", total_size);
    printf("Correct predictions: %f\n", total_correct);
    printf("Porcentage of correct predictions: %f\%\n", (total_correct / total_size) * 100);
    //Add the percentage of correct predictions to the file
    FILE* file = fopen("results.txt", "a");
    if (file == NULL) {
        err(EXIT_FAILURE, "Error opening file\n");
    }
    fprintf(file, "\n%s | Total number of image : %f\n", __TIME__, total_size);
    fprintf(file, "%s | Correct predictions: %f\n",__TIME__, total_correct);
    fprintf(file, "%s | Porcentage of correct predictions: %f\%\n", __TIME__, (total_correct / total_size) * 100);
    fclose(file);
    return total_correct;
}


int main() {
    srand(time(NULL));

    FILE* file = fopen("results.txt", "a");
    if (file == NULL) {
        err(EXIT_FAILURE, "Error opening file\n");
    }
    fprintf(file, "\n%s | New Start of program\n", __TIME__);
    fclose(file);

    int num_layers = 3;
    int layer_sizes[] = {INPUT_SIZE, 100, NUM_CLASSES}; // Example for MNIST dataset
    NeuralNetwork* nn = create_neural_network(num_layers, layer_sizes);

    // Import des poids des perceptrons si le fichier existe

    int correct = 0;
    while (1) {
        correct = 0;

        correct += test_model(nn, "/home/clement.forget/NeuralNetwork/data/testing_data/", 1000);

        // Entraîner le modèle
        start_train(nn);
    }



    free_neural_network(nn);

    return 0;
}

/*
 Example usage
int main() {
    srand(time(NULL));

    int num_layers = 4;
    int layer_sizes[] = {784, 128, 64, 10}; // Example for MNIST dataset
    NeuralNetwork* nn = create_neural_network(num_layers, layer_sizes);

    // Here you would load your training data
    // float** inputs = load_inputs();
    // float** targets = load_targets();
    // int num_samples = get_num_samples();

    // train(nn, inputs, targets, num_samples, 100, 0.01);

    // Example prediction
    float input[784] = {0}; // Initialize with your input data
    predict(nn, input);

    free_neural_network(nn);
    return 0;
}
*/

