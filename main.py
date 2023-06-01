import functions
import os

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

path = 'images/*'

# Gan function to upscale image by 4x
functions.gan("cuda", 'models/RRDB_ESRGAN_x4.pth', path)
# Tessaract function to read text from image


# Get all file names in the folder
file_names = os.listdir('images/')
# Filter PNG files
png_files = [filename for filename in file_names if filename.endswith(".png")]

# Print the names of PNG files
for png_file in png_files:
    text = functions.tesseractFunction("images/" + png_file)
    print(png_file)
    print(text + "\n")
    file_path = "results/"+png_file+".txt"
    with open(file_path, "w") as file:
      file.write(text)
