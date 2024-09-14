#ifndef TRAIN_H
#define TRAIN_H

#include <stddef.h>  // Pour size_t

// Définition des constantes
#define INPUT_SIZE 32*32  // Taille des images (32x32 pixels#define NUM_CLASSES 26    // 26 classes pour A-Z
#define BATCH_SIZE 64     // Taille du batch
#define MAX_IMAGES 100  // Taille maximale du dataset
#define NUM_CLASSES 36

// Structure pour stocker une image et son label
struct ImageData{
    float* pixels;  // Image normalisée (pixels)
    int label;                 // Label de l'image (0-25 correspondant à A-Z)
} ;

// Prototypes des fonctions

/**
 * Charge une image depuis un fichier et la normalise en niveaux de gris.
 *
 * @param filepath Chemin du fichier image
 * @param output_pixels Tableau pour stocker les pixels normalisés
 * @param target_width Largeur cible de l'image (doit être 32)
 * @param target_height Hauteur cible de l'image (doit être 32)
 * @return 1 si l'image est chargée avec succès, 0 sinon
 */
float* load_png_image(const char* file_path, int* width, int* height, int* channels);

/**
 * Charge toutes les images d'un répertoire et les associe à leurs labels respectifs.
 *
 * @param directory Chemin du répertoire contenant les sous-dossiers des classes
 * @param dataset Tableau où stocker les images chargées
 * @param dataset_size Pointeur vers un entier pour stocker la taille du dataset
 * @return 1 si le chargement réussit, 0 sinon
 */
int load_images_from_directory(const char* directory, struct ImageData* dataset);

/**
 * Sélectionne un ensemble d'images aléatoires dans le dataset.
 *
 * @param dataset Dataset complet d'images
 * @param dataset_size Taille du dataset complet
 * @param selected_images Tableau pour stocker les images sélectionnées
 * @param num_images Nombre d'images à sélectionner
 */
void select_random_images(struct ImageData* dataset, int dataset_size, struct ImageData* selected_images, int num_images);

void create_one_hot_labels(float** labels_matrix, struct ImageData* dataset, int num_samples);

/**
 *Fonction pour initialiser tout les perceptrons
 *
 */
void init_perceptrons();

/**
 * Fonction pour calculer les probabilités softmax pour chaque perceptron.
 *
 * @param perceptrons Tableau de perceptrons
 * @param x Entrée des caractéristiques
 * @param ze Tableau pour stocker les résultats softmax
 */
void z_perceptrons(struct Perceptron* perceptrons, float* x, float* ze);

/**
 * Fonction pour calculer la perte de classification softmax.
 *
 * @param perceptrons Tableau de perceptrons
 * @param x_tab Matrice des caractéristiques d'entrée
 * @param y_tab Matrice des labels (one-hot encoding)
 * @param n_samples Nombre d'échantillons
 * @return La perte softmax
 */
float loss_classif_softmax(struct Perceptron* perceptrons, float** x_tab, float** y_tab, size_t n_samples);

/**
 * Effectue la rétropropagation avec softmax pour la classification.
 *
 * @param perceptrons Tableau de perceptrons
 * @param x_tab Matrice des caractéristiques d'entrée
 * @param y_tab Matrice des labels (one-hot encoding)
 * @param n_samples Nombre d'échantillons
 * @param eta Taux d'apprentissage
 */
void retropopagation_classif_softmax(struct Perceptron* perceptrons, float** x_tab, float** y_tab, size_t n_samples, float eta);

/**
 * Sélectionne un batch aléatoire de données d'entraînement.
 *
 * @param x_tot Ensemble d'entrainement
 * @param y_tot Labels d'entraînement
 * @param total_size Taille totale de l'ensemble d'entraînement
 * @param x_tab Batch de données d'entrée
 * @param y_tab Batch de labels
 * @param batch_size Taille du batch
 */
void selection_bash(float** x_tot, float** y_tot, size_t total_size, float** x_tab, float** y_tab, int batch_size);

/**
 * Fonction principale d'entraînement.
 *
 * @param entrainement_imgs Matrice des images d'entraînement
 * @param entrainement_chiffres Matrice des labels d'entraînement
 * @param total_size Nombre total d'exemples dans l'ensemble d'entraînement
 */
void train(float** entrainement_imgs, float** entrainement_chiffres, size_t total_size);

#endif  // TRAIN_H

