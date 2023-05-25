import os
import functions
import numpy as np
import matplotlib.pyplot as plt

# Define the folder path
folder_path = "test_dataset/images/"

# Get a list of all files and directories in the folder
items = os.listdir(folder_path)

# Run the GAN model on the images
#gan("cuda", 'models/RRDB_ESRGAN_x4.pth', 'test_dataset/images/*')


# Create arrays for each metric
lowJaroArr = []
highJaroArr = []
jaroImprovementArr = [] 
psnrArr = []
levenshteinLowArr = []
levenshteinHighArr = []
levenshteinImprovementArr = []
cosineLowArr = []
cosineHighArr = []
cosineImprovementArr = []


# Specify the folder path
folder_pathAux = 'simulated-sources/images/'
functions.gan("cuda", 'models/RRDB_ESRGAN_x4.pth', 'simulated-sources/images/*')



# Loop through the files in the folder
for filename in os.listdir(folder_pathAux):
    # Check if the file has a .png extension
    if filename.endswith('.png'):
        # Get the file name without extension
        name_without_extension = os.path.splitext(filename)[0]
        arr = functions.metricsResults(functions.tesseractFunction(folder_pathAux + name_without_extension + '.png'), functions.tesseractFunction('results2/' + name_without_extension + '.png'), functions.extract_text_from_file("simulated-sources/ground_truth" + name_without_extension + '.txt'))
        # Print the file name without extension
        lowJaroArr.append(arr[0])
        highJaroArr.append(arr[1])
        jaroImprovementArr.append(arr[2])
        psnrArr.append(arr[3])
        levenshteinLowArr.append(arr[4])
        levenshteinHighArr.append(arr[5])
        levenshteinImprovementArr.append(arr[6])
        cosineLowArr.append(arr[7])
        cosineHighArr.append(arr[8])
        cosineImprovementArr.append(arr[9])
        print(arr)


# # Loop through each item and check if it is a file
# for item in items:
#     if os.path.isfile(os.path.join(folder_path, item)) and item.endswith(".png"):
#         name, extension = os.path.splitext(item)
#         print(name)
#         arr = metricsResults(tesseractFunction('test_dataset/images/' + name + '.png'), tesseractFunction('results/' + name + '.png'), jsonText(name))
#         lowJaroArr.append(arr[0])
#         highJaroArr.append(arr[1])
#         jaroImprovementArr.append(arr[2])
#         psnrArr.append(arr[3])
#         levenshteinLowArr.append(arr[4])
#         levenshteinHighArr.append(arr[5])
#         levenshteinImprovementArr.append(arr[6])
#         cosineLowArr.append(arr[7])
#         cosineHighArr.append(arr[8])
#         cosineImprovementArr.append(arr[9])
#         print(arr)

# Calculate the average of each metric
lowJaroAvg = np.average(lowJaroArr)
highJaroAvg = np.average(highJaroArr)
jaroImprovementAvg = np.average(jaroImprovementArr)
psnrAvg = np.average(psnrArr)
levenshteinLowAvg = np.average(levenshteinLowArr)
levenshteinHighAvg = np.average(levenshteinHighArr)
levenshteinImprovementAvg = np.average(levenshteinImprovementArr)
cosineLowAvg = np.average(cosineLowArr)
cosineHighAvg = np.average(cosineHighArr)
cosineImprovementAvg = np.average(cosineImprovementArr)

# Print the average of each metric
print("Average Jaro Winkler distance between the texts (high quality image to original):" + str(highJaroAvg))
print("Average Jaro Winkler distance between the texts (low quality image to original):" + str(lowJaroAvg))
print("Average Jaro Winkler distance improvement: " + str(jaroImprovementAvg))
print("Average PSNR (Peak to Signal Noise Ratio) of the images is: " + str(psnrAvg))
print("Average Levenshtein distance between the texts (low quality image): " + str(levenshteinLowAvg))
print("Average Levenshtein distance between the texts (high quality image): " + str(levenshteinHighAvg))
print("Average Levenshtein improvement: " + str(levenshteinImprovementAvg))
print("Average cosine similarity between the texts (low quality image) is: " + str(cosineLowAvg))
print("Average cosine similarity between the texts (high quality image) is: " + str(cosineHighAvg))
print("Average cosine similarity improvement: " + str(cosineImprovementAvg))

# Plot the average Jaro Winkler distance against the original text
labels = ['Low image', 'High image']
plt.bar(labels, [lowJaroAvg, highJaroAvg])
plt.xlabel('Values')
plt.ylabel('Counts')
plt.title('Comparison of Jaro Winkler distance averages against the original text')
plt.show()

# Plot the average levenshtein distance against the original text
labels = ['Low image', 'High image']
plt.bar(labels, [levenshteinLowAvg, levenshteinHighAvg])
plt.xlabel('Values')
plt.ylabel('Counts')
plt.title('Comparison of Levenshtein distance averages against the original text')
plt.show()

# Plot the average cosine similarity against the original text
labels = ['Low image', 'High image']
plt.bar(labels, [cosineLowAvg, cosineHighAvg])
plt.xlabel('Values')
plt.ylabel('Counts')
plt.title('Comparison of cosine similarity averages against the original text')
plt.show()




# # Extract texts from ground truth, low resolution image and high resolution image
# original = jsonText('83635935')
# tessLowW = tesseractFunction('test.png')
# tessHighW = tesseractFunction('srgan.png')
