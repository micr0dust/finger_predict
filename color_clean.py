import os
import cv2
from tqdm import tqdm
import numpy as np

def remove_noise(image):
    # Define a kernel for the morphological operations
    kernel = np.ones((2,2),np.uint8)

    # Perform opening operation (erosion followed by dilation)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return opening

def clear_pixels_by_hue(image_path, output_image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask
    mask = (hsv_image[:, :, 0] > 15) | (hsv_image[:, :, 0] < 0)

    # Clear the pixels where the mask is True
    image[mask] = 0

    cv2.imwrite(output_image_path, remove_noise(image))

for i in tqdm(range(int('100000',2))):
    # Define the input and output directories
    folder = "{0:05b}".format(i)
    input_dir = f'./processed_data_right/{folder}/'
    output_dir = f'./hue_data/{folder}/'
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Loop through all the files in the input directory
    for filename in os.listdir(input_dir):
        clear_pixels_by_hue(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

