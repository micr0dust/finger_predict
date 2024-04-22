import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image, title='Image'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
    # plt.savefig(f'img/red_mask/{title}.png')

# Load the image
image = cv2.imread('./processed_data_left/00000/00000_L_0.jpg')

def printHSV(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram of the hue channel
    hue_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])

    # Plot the color distribution
    plt.figure(figsize=(12, 6))

    plt.bar(range(180), hue_hist.flatten())
    plt.xlabel('Hue Value')
    plt.ylabel('Pixel Count')
    plt.title('Hue Distribution')

    plt.tight_layout()
    plt.show()


def printRGB(image):
    # 假設 img 是一個 RGB 圖像
    r, g, b = cv2.split(image)

    plt.figure(figsize=(10, 6))

    plt.hist(r.ravel(), bins=256, color='red', alpha=0.5, label='Red')
    plt.hist(g.ravel(), bins=256, color='green', alpha=0.5, label='Green')
    plt.hist(b.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')

    plt.title('Color Distribution')
    plt.xlabel('Color Value')
    plt.ylabel('Pixel Count')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def mask_red_channel(image, lower_bound, upper_bound):
    # 分割圖像的紅色、綠色和藍色通道
    r, g, b = cv2.split(image)

    # 創建一個遮罩，將紅色通道中不在指定範圍內的值設為 0
    mask = (g < lower_bound) | (g > upper_bound)
    r[mask] = 0
    g[mask] = 0
    b[mask] = 0

    # 將分割的通道重新組合成一個圖像
    masked_image = cv2.merge([r, g, b])

    return masked_image

def red_channel_only(image):
    image=image[:, :, 2]
    return image



# for i in range(0, 256-50, 50):
#     masked_image = mask_red_channel(image, lower_bound=i, upper_bound=i+50)
#     plot_image(masked_image, title=f'Red Channel Mask ({i}~{i+50})')
image=red_channel_only(image)
plot_image(image)
