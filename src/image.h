//
// Created by clement on 01/10/24.
//

#ifndef IMAGE_H
#define IMAGE_H

struct ImageData{
    float* pixels;  // Image normalisée (pixels)
    int label;                 // Label de l'image (0-25 correspondant à A-Z)
    int width;
    int height;
} ;

int is_directory(const char* path);

int is_regular_file(const char* path);

void select_random_images(struct ImageData* dataset, int dataset_size, struct ImageData* selected_images, int num_images);

void create_one_hot_labels(float** labels_matrix, struct ImageData* dataset, int num_samples);

#endif //IMAGE_H
