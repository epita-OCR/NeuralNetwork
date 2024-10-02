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
                char *dir[1];
                get_filename_without_extension(directory_path, dir);
                dataset[count].label = ((char) dir[0]) - 97; // Définir ici la bonne étiquette en fonction du nom de fichier ou du répertoire
                //printf("FilePath : %s || Domc name = %s || n = %i\n", filepath, dir, ((char) dir[0]) - 97);
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


void start_train(NeuralNetwork *nn) {
    printf("Hey\n");
    printf("Allocating %lu bytes for dataset\n", MAX_IMAGES * sizeof(struct ImageData));

    printf("Initializing perceptrons\n");

    // Charger les images du répertoire "Train". Les images sont dans des sous-dossiers. Il faur donc parcourir les sous-dossiers pour charger les images.
    char* directory_path = "/home/clement/Documents/refineddataset/dataset/";
    // Parcourir les sous-dossiers pour charger les images
    char* subdirectories[] = {"a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};

    int dataset_size = 0;
    //pid_t pids[NUM_CLASSES];

    for (int i = 0; i < NUM_CLASSES; i++) {
        //if ((pids[i] = fork()) == 0) {
            //Child process
            char dir[256];
            snprintf(dir, sizeof(dir), "%s%s", directory_path, subdirectories[i]);

            struct ImageData *dataset = calloc(MAX_IMAGES, sizeof(struct ImageData));
            if (dataset == NULL) {
                err(1, "Memory allocation failed\n");
            }

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
            for (int i = 0; i < MAX_IMAGES; i++) {
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

            printf("Creating one-hot labels for training\n");
            create_one_hot_labels(y_train, selected_images, MAX_IMAGES);
            printf("Training the model\n");
            train(nn, x_train, y_train, dataset_size, 100, 0.1);

            // Libérer la mémoire allouée
            for (int i = 0; i < 60000; i++) {
                free(y_train[i]);
            }
            free(x_train);
            free(y_train);
            free(dataset);


           // exit(0);  // Child process exits
        }
    }

    // Parent process waits for all child processes to complete
    //for (int i = 0; i < NUM_CLASSES; i++) {
     //   waitpid(pids[i], NULL, 0);
    //}
//}

//Test the model
int test_model(NeuralNetwork *nn, const char* directory_path, int num_images) {

    // Charger les images du répertoire "Validation". Les images sont dans des sous-dossiers. Il faur donc parcourir les sous-dossiers pour charger les images.
    //char* directory_path = "/home/clement/Documents/archive/Validation/";
    // Parcourir les sous-dossiers pour charger les images
    char* subdirectories[] = {"a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
    float correct = 0;
    float total_size = 0;
    for (int letter = 0; letter < NUM_CLASSES; letter++) {
        char* dir[39] = { 0 };
        strcat(dir, directory_path);
        strcat(dir, subdirectories[letter]);

        int dataset_size = 0;

        struct ImageData *dataset = calloc(MAX_IMAGES, sizeof(struct ImageData));
        if (dataset == NULL) {
            err(1, "Memory allocation failed\n");

        }

        printf("Loading images from directory %s\n", dir);
        dataset_size = load_images_from_directory(dir, dataset);
        printf("dataset_size = %i\n", dataset_size);
        total_size += dataset_size;

        //printf("Nombre total d'images chargées : %d\n", dataset_size);
        // Use predict function to predict the label of the image
        for (int i = 0; i < dataset_size; i++) {
            float* input = dataset[i].pixels;

            int predicted_label = predict(nn, input);
            //printf("For label %i, predicted label = %i\n", dataset[i].label, predicted_label);
            if (predicted_label == dataset[i].label) {
                correct++;
            }
        }
    }
    printf("Number total of images : %f\n", total_size);
    printf("Correct predictions: %f\n", correct);
    printf("Porcentage of correct predictions: %f\%\n", (correct / total_size) * 100);
    //Add the percentage of correct predictions to the file
    FILE* file = fopen("results.txt", "a");
    if (file == NULL) {
        err(EXIT_FAILURE, "Error opening file\n");
    }
    fprintf(file, "\n%s | Total number of image : %f\n", __TIME__, total_size);
    fprintf(file, "%s | Correct predictions: %f\n",__TIME__, correct);
    fprintf(file, "%s | Porcentage of correct predictions: %f\%\n", __TIME__, (correct / total_size) * 100);
    fclose(file);
    return correct;
}


int main() {
    srand(time(NULL));

    FILE* file = fopen("results.txt", "a");
    if (file == NULL) {
        err(EXIT_FAILURE, "Error opening file\n");
    }
    fprintf(file, "\n%s | New Start of program\n", __TIME__);
    fclose(file);

    int num_layers = 4;
    int layer_sizes[] = {INPUT_SIZE, 128, 64, NUM_CLASSES}; // Example for MNIST dataset
    NeuralNetwork* nn = create_neural_network(num_layers, layer_sizes);

    // Import des poids des perceptrons si le fichier existe

    int correct = 0;
    while (correct < 6000) {
        correct = 0;

        correct += test_model(nn, "/home/clement/Documents/refineddataset/dataset/", 1000);

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

