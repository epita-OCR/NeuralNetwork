#include <criterion/criterion.h>
#include <criterion/logging.h>
#include "../src/train.h"  // Assurez-vous que ce fichier contient la déclaration de load_image
#include "image_tests.h"
#define TARGET_WIDTH 32
#define TARGET_HEIGHT 32

Test(load_image, valid_image) {
    float output_pixels[TARGET_WIDTH * TARGET_HEIGHT];
    int result = load_image("/home/clement/Documents/test2/archive/dataset/Train/&/__0_7728.png", output_pixels, TARGET_WIDTH, TARGET_HEIGHT);
    cr_assert(result == 1, "Expected load_image to return 1 for a valid image");
}

Test(load_image, invalid_image_path) {
    float output_pixels[TARGET_WIDTH * TARGET_HEIGHT];
    int result = load_image("/home/clement/Picures/Screenshots/Screenshotfrom2024-06-11 20-20-37.png", output_pixels, TARGET_WIDTH, TARGET_HEIGHT);
    cr_assert(result == 0, "Expected load_image to return 0 for an invalid image path");
}

Test(load_image, invalid_image_size) {
    float output_pixels[TARGET_WIDTH * TARGET_HEIGHT];
    int result = load_image("/home/clement/Pictures/Screenshots/Screenshotfrom2024-06-11 20-20-37.png", output_pixels, TARGET_WIDTH, TARGET_HEIGHT);
    cr_assert(result == 0, "Expected load_image to return 0 for an image with invalid size");
}

Test(load_image, invalid_image_channels) {
    float output_pixels[TARGET_WIDTH * TARGET_HEIGHT];
    int result = load_image("/home/clement/Pictures/Screenshots/Screenshotfrom2024-06-11 20-20-37.png", output_pixels, TARGET_WIDTH, TARGET_HEIGHT);
    cr_assert(result == 0, "Expected load_image to return 0 for an image with invalid channels");
}

