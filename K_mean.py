import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set the path to the directory containing the images
PATH = './processed_data_all/'

# Get the list of image file names in the directory
image_files = []
for folder in range(int('100000',2)):
    image_dir = PATH+"{0:05b}/".format(folder)
    image_files += [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Load and preprocess the images
images = []
for file in image_files:
    image_path = os.path.join(PATH+file.split('_')[0], file)
    image = cv2.imread(image_path)
    # Preprocess the image if needed (e.g., resize, normalize, etc.)
    # ...
    images.append(image)
# Convert the list of images to a numpy array
images = np.array(images)

# Flatten the images to a 2D array
flattened_images = images.reshape(images.shape[0], -1)

# Define the number of clusters
num_clusters = 5

# Create a K-means model and fit it to the flattened images
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(flattened_images)

# Get the cluster labels for each image
labels = kmeans.labels_

# Print the cluster labels for each image
for i, file in enumerate(image_files):
    print(f"Image: {file}, Cluster: {labels[i]}")



from sklearn.decomposition import PCA

# Use PCA to reduce the dimensionality of the data to 2D
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(flattened_images)

# Perform K-means clustering on the reduced data
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(reduced_data)

# Plot the data points and color them by their cluster
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis')

# Plot the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.show()