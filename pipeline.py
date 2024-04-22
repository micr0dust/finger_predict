import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

# 縮放圖片
def resize(image, size):
    height, width = image.shape[:2]
    if height > width:
        new_dimensions = (int(width * size / height), size)
    else:
        new_dimensions = (size, int(height * size / width))
    return cv2.resize(image, new_dimensions, interpolation = cv2.INTER_AREA)
    
# 先腐蝕再膨脹清除小的白塊雜訊
def remove_noise(image):
    # Define a kernel for the morphological operations
    kernel = np.ones((2,2),np.uint8)

    # Perform opening operation (erosion followed by dilation)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return opening

# 透過色相清除不需要的像素
def clear_pixels_by_hue(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create a mask
    mask = (hsv_image[:, :, 0] > 16)
    
    # Clear the pixels where the mask is True
    image[mask] = 0
    return image

# 只取紅色通道
def red_channel_only(image):
    image=image[:, :, 2]
    return image

# 處理單張圖片
def pipeline_process(image):
    # image = resize(image, 64)
    image = red_channel_only(image)
    # image = remove_noise(image)
    return image

# 整個資料夾批次處理
def pipeline_folder(input_path, output_path):
    for i in tqdm(range(int('100000',2))):
        # Define the input and output directories
        folder = "{0:05b}".format(i)
        input_dir = f'{input_path}/{folder}/'
        output_dir = f'{output_path}/{folder}/'
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Loop through all the files in the input directory
        for filename in os.listdir(input_dir):
            image = cv2.imread(os.path.join(input_dir, filename))
            cv2.imwrite(os.path.join(output_dir, filename), pipeline_process(image))


# ----- 以下為使用範例 -----------------------------------------
if __name__ == "__main__":
    PATH = "./processed_data_all/"
    DIR = "00000"

    # 整個資料夾批次處理
    pipeline_folder(PATH, "./pipelined_data_all")

    # # Read the image
    # files = os.listdir(PATH+DIR)
    # image = cv2.imread(os.path.join(PATH+DIR, files[random.randint(0, len(files)-1)]))

    # # Process the image
    # processed_image = pipeline_process(image)

    # # Plot the image
    # plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()