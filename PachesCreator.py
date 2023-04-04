import os
import numpy as np
import tifffile as tiff
# Set input and output directories
input_dir = 'C:/Users/john_/OneDrive/Desktop/mito/'
output_dir_images = 'C:/Users/john_/OneDrive/Desktop/mito/generated_patches/images/'
output_dir_masks = 'C:/Users/john_/OneDrive/Desktop/mito/generated_patches/masks/'

# Load training images and ground truth masks
training_data = tiff.imread(os.path.join(input_dir, 'training.tif'))
ground_truth_data = tiff.imread(os.path.join(input_dir, 'training_groundtruth.tif'))
# Set patch size
patch_size = 256
# Loop through each slice of the training and ground truth data
for i in range(training_data.shape[0]):
    # Crop training image to patches
    for x in range(0, training_data.shape[1], patch_size):
        for y in range(0, training_data.shape[2], patch_size):
            patch = training_data[i, x:x+patch_size, y:y+patch_size]
            if patch.shape == (patch_size, patch_size):
                # Save training patch
                tiff.imsave(os.path.join(output_dir_images, f'image_{i}_{x}_{y}.tif'), patch)

                # Crop corresponding ground truth mask patch
                mask_patch = ground_truth_data[i, x:x+patch_size, y:y+patch_size]
                if mask_patch.shape == (patch_size, patch_size):
                    # Save ground truth mask patch
                    tiff.imsave(os.path.join(output_dir_masks, f'mask_{i}_{x}_{y}.tif'), mask_patch)

