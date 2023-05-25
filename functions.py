import numpy as np
import cv2
import pytesseract
import jaro
import json
import os.path as osp
import glob
import torch
import RRDBNet_arch as arch
from PIL import Image
from Levenshtein import distance as lev
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from text file
def extract_text_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return ""
    except IOError:
        print(f"Error reading file '{file_path}'.")
        return ""

# Function to call ESRGAN API from replicate
def apiESRGAN(imagePATH):
    output = replicate.run(
        "xinntao/realesrgan:1b976a4d456ed9e4d1a846597b7614e79eadad3032e9124fa63859db0fd59b56",
        input={"img": open(imagePATH, "rb")}
    )
    return output

# Function to extract text from json file
def jsonText(name):
    with open('test_dataset/ground_truth/' + name + '.json', 'r') as f:
        data = json.load(f)
    text_list = []
    for item in data['form']:
        text_list.append(item['text'])
    text = ' '.join(text_list)
    return text

# Function to calculate the jaro distance between two texts
def jaroMetric(text1, text2):
    return jaro.jaro_winkler_metric(text1, text2)

# Function to extract text from image using tesseract
def tesseractFunction(imgPath):
    text = pytesseract.image_to_string(Image.open(imgPath))
    text = text.replace('\n', ' ')
    return text

# Function to calculate the PSNR between two images
def psnr(img1, img2):
    # read the two images
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    # resize the images to the same dimensions
    img1_resized = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    # convert the images to grayscale
    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # calculate the MSE (mean squared error)
    mse = np.mean((img1_gray - img2_gray) ** 2)
    # calculate the maximum pixel value
    max_pixel_value = np.max(img1_gray)
    # calculate the PSNR
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

# Function to calculate the levenstein distance between two texts
def levenshtein(img1, original):
    return lev(img1, original)

# Function to calculate the cosine similarity between two texts
def cosineSimilarity(img1, original):
    # Convert the texts to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([img1, original])
    # Calculate cosine similarity between the vectors
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

def textExtract(lowquality, highquality, originalName):
    # Extract texts from ground truth, low resolution image and high resolution image
    original = jsonText(originalName)
    tessLowW = tesseractFunction(lowquality)
    tessHighW = tesseractFunction(highquality)
    textArray = [original, tessLowW, tessHighW]
    return textArray

def metricsResults(lowquality, highquality, original):
    # Calculate the jaro distance between the texts
    lowJaro = jaroMetric(lowquality, original)
    highJaro = jaroMetric(highquality, original)
    jaroImprovement = highJaro - lowJaro
    # print("Jaro Winkler distance between the texts (high quality image to original):" + str(highJaro))
    # print("Jaro Winkler distance between the texts (low quality image to original):" + str(lowJaro))
    # print("Jaro Winkler distance improvement: " + str(jaroImprovement))

    # Calculate the PSNR
    psnrResult = psnr('test.png', 'srgan.png')
    # print("The PSNR (Peak to Signal Noise Ratio) of the images is: " + str(psnrResult))

    # Calculate the Levenshtein distance between the texts
    max_distance = max(len(lowquality), len(original))
    levenshteinLow = (levenshtein(lowquality, original) / max_distance) 
    levenshteinHigh = (levenshtein(highquality, original) / max_distance) 

    # levenshteinLow = levenshtein(lowquality, original)
    # levenshteinHigh = levenshtein(highquality, original)
    levenshteinImprovement = levenshteinLow - levenshteinHigh
    # print("Levenshtein distance between the texts (low quality image): " + str(levenshteinLow))
    # print("Levenshtein distance between the texts (high quality image): " + str(levenshteinHigh))
    # print("Levenshtein improvement: " + str(levenshteinImprovement))

    # Calculate the cosine similarity between the texts
    cosineLow = cosineSimilarity(lowquality, original)
    cosineHigh = cosineSimilarity(highquality, original)
    cosineImprovement = cosineHigh - cosineLow
    # print("The cosine similarity between the texts (low quality image) is: " + str(cosineLow))
    # print("The cosine similarity between the texts (high quality image) is: " + str(cosineHigh))
    # print("Cosine similarity improvement: " + str(cosineImprovement))
    
    # Create array with all the metrics
    metricsArray = [lowJaro, highJaro, jaroImprovement, psnrResult, levenshteinLow, levenshteinHigh, levenshteinImprovement, cosineLow, cosineHigh, cosineImprovement]
    return metricsArray

# Function to run GAN model on images
def gan(device, model_path, test_img_folder):
    Choose_device = device 
    model_path = model_path
    device = torch.device(Choose_device) 
    test_img_folder = test_img_folder

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        print("Model output:", output.shape)
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        print("Output transposed:", output.shape)
        output = (output * 255.0).round()
        print("Output rounded:", output.shape)
        cv2.imwrite('results/{:s}.png'.format(base), output)
        print("Image written to disk")
