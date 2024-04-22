import os
import cv2
from tqdm import tqdm

def resize_image(input_image_path, output_image_path, size):
    original_image = cv2.imread(input_image_path)
    height, width = original_image.shape[:2]
    if height > width:
        new_dimensions = (int(width * size / height), size)
    else:
        new_dimensions = (size, int(height * size / width))
    resized_image = cv2.resize(original_image, new_dimensions, interpolation = cv2.INTER_AREA)
    cv2.imwrite(output_image_path, resized_image)

for i in tqdm(range(int('100000',2))):
    # Define the input and output directories
    folder = "{0:05b}".format(i)
    input_dir = f'./raw_data_right/{folder}/'
    output_dir = f'./processed_data_right/{folder}/'
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Loop through all the files in the input directory
    for idx, filename in enumerate(os.listdir(input_dir)):
        resize_image(os.path.join(input_dir, filename), os.path.join(output_dir, f"{folder}_L_{idx}.jpg"), 64)
    