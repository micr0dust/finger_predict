import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = []
labels = []

PATH = 'processed_data_all'
for folder in tqdm(os.listdir(PATH),desc='load image'):
    if not os.path.isdir(f'{PATH}/{folder}'):
        continue
    for file in os.listdir(f'{PATH}/{folder}'):
        image = cv2.imread(f'{PATH}/{folder}/{file}')
        processed_image = image
        data.append(processed_image)
        labels.append(folder)

X = np.array(data)/255.0

# 將圖像數據重塑為二維數據
X = X.reshape(X.shape[0], -1)

pca = PCA(n_components=300)
X_reduced = pca.fit_transform(X)

def var_explain():
    # 繪製每個主成分的解釋變異數比例
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(explained_variance_ratio_cumsum)), explained_variance_ratio_cumsum)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.show()

def plot_pca_recovered(X_reduced):
    # 將降維後的數據轉換回原始的像素空間
    X_recovered = pca.inverse_transform(X_reduced)

    # 選擇要視覺化的圖像索引
    image_index = 0

    # 重新讀取選擇的圖像
    image = data[image_index]

    # 將數據轉換回原始的圖像形狀
    recovered_image = X_recovered[image_index].reshape(image.shape)

    # 繪製原始圖像和恢復的圖像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Original Image: {X.shape[1]}')

    plt.subplot(1, 2, 2)
    plt.imshow(recovered_image, cmap='gray')
    plt.title(f'Reduced Image: {X_reduced.shape[1]}')

    plt.show()

# var_explain()
plot_pca_recovered(X_reduced)