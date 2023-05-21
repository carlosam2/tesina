# import tesst

# This program uses an already trained GAN model to upscale an image by 4x and then uses Tesseract to read text from the upscaled image
# The recommended use for this program is to use it to extract text from a low resolution image or degraded document

# Warning:
# This program requires a GPU to run the GAN model locally
# If you do not have a GPU, you can use Google Colab to run the program
# The notebook is located in the repository as "OCR.ipynb"

# If you don't have a GPU, but still want to run the program locally, you can change the device to "cpu" in the gan function
# This will run the GAN model on your CPU, but it will take a very long time to upscale the image

# Instructions:
# If there is no images folder, create one (This is where the image to be upscaled will be saved)
# If there is no results folder, create one (This is where the upscaled image will be saved)
# Add image to images folder or change path to image (The image must be a .png file)
# Run main.py to upscale image by 4x and read text from image 

# Output:
# Upscaled image will be saved to results folder
# Text from image will be printed to console

# path = 'images/*'

# # Gan function to upscale image by 4x
# tesst.gan("cuda", 'models/RRDB_ESRGAN_x4.pth', path)
# # Tessaract function to read text from image
# text = tesst.tesseractFunction('test.png')
# print(text + "\n")


import os
import shutil

def copy_png_files(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through the files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file has a .png extension
        if filename.endswith('.txt'):
            # Create the source and destination file paths
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            # Copy the file to the destination folder
            shutil.copy2(source_file, destination_file)

# Example usage
source_folder = 'simulated-sources/best-poetry/'
destination_folder = 'simulated-sources/ground_truth/'

# copy_png_files(source_folder, destination_folder)

import os

def remove_part_of_name(folder_path, part_to_remove):
    # Loop through the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .png extension
        if filename.endswith('.png'):
            # Remove the specified part from the file name
            new_filename = filename.replace(part_to_remove, '')

            # Create the source and destination file paths
            source_file = os.path.join(folder_path, filename)
            destination_file = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(source_file, destination_file)

# Example usage
folder_path = 'simulated-sources/images/'
part_to_remove = '-simulated-60dpi'

remove_part_of_name(folder_path, part_to_remove)
