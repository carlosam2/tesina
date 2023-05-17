import tesst

# This program uses an already trained GAN model to upscale an image by 4x and then uses Tesseract to read text from the upscaled image
# The recommended use for this program is to use it to extract text from a low resolution image or degraded document

# Warning:
# This program requires a GPU to run the GAN model locally
# If you do not have a GPU, you can use Google Colab to run the program
# The notebook is located in the repository as "OCR.ipynb"

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
tesst.gan("cuda", 'models/RRDB_ESRGAN_x4.pth', path)
# Tessaract function to read text from image
text = tesst.tesseractFunction('test.png')
print(text + "\n")

