# %%
import os
from glob import iglob
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# importing or loading the dataset
dataset = pd.read_csv('20240905_175121.csv', names=["files", "rating"])
path = 'frontalimages_manuallyaligne_greyscale/'
#path = 'test'
# %%
dataset['files']
# %%
faces_list = []

for file in dataset['files']:
    img=imread(path+'/'+file)
    dim = img.shape
    face = pd.Series(img.flatten(),name=path)
    faces_list.append(face)

# Convert list of flattened images to a numpy array
faces = np.array(faces_list)

# Compute the average image
avg = np.mean(faces, axis=0)

# Subtract the average image from each image
faces = faces -avg

# %%
#Fits with PCA
pca = PCA()
pca.fit(faces)

components = pca.components_
print(components)
# Visually present the first few PCA components as images
pca_transformed = pca.transform(faces)

plt.figure(figsize=[15,8])
for i in range (0,3):

    # Select the PCA component to analyze 
    component_index = i
    pca_component = pca_transformed[:, component_index]
    # Find the images with the maximum and minimum projection scores for the component
    max_score_index = np.argmax(pca_component)
    min_score_index = np.argmin(pca_component)

    # Reconstruct images 
    def reconstruct(index, n_components=1):
        # Keep only the first n_components, set the rest to 0
        pca_keep = np.zeros_like(pca_transformed)
        pca_keep[:, :n_components] = pca_transformed[:, :n_components]
        return (pca.inverse_transform(pca_keep[index])+avg).reshape(img.shape)

    max_reconstructed = reconstruct(max_score_index, n_components=1)
    min_reconstructed = reconstruct(min_score_index, n_components=1)

    # Visualize the min, mean, and max images
    # Min Image
    plt.subplot(3, 3, 3*i + 1)  # 3 rows, 3 columns, fill in order
    plt.imshow(min_reconstructed, cmap='gray')
    plt.title(f"Min Comp {component_index}")
    plt.axis('off')
    
    # Mean Image
    plt.subplot(3, 3, 3*i + 2)
    plt.imshow(avg.reshape(img.shape), cmap='gray')
    plt.title(f"Mean Image")
    plt.axis('off')

    # Max Image
    plt.subplot(3, 3, 3*i + 3)
    plt.imshow(max_reconstructed, cmap='gray')
    plt.title(f"Max Comp {component_index}")
    plt.axis('off')
plt.savefig('PCA_component.jpg')
plt.show()

# %%
# Analysis
# Get explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
# Compute cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
# Number of components explaining 95% of the variance
k = np.argmax(cumulative_variance*100 > 80) + 1
print("The number of components chosen(explaining 80% variance): ", k)

# Plot the explained variance ratio
plt.figure(figsize=(25, 6))
plt.bar(range(0, len(explained_variance_ratio)), explained_variance_ratio, color='skyblue')
# Line plot for cumulative variance
plt.plot(range(0, len(cumulative_variance)), cumulative_variance, color='red', marker='o', label='Cumulative Variance')


plt.xlabel('Principal Component')
plt.ylabel('Variance Explained Ratio')
plt.title('Variance Explained by Each Principal Component')
x_ticks = np.arange(0, len(explained_variance_ratio) , 5)
plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=45, ha='right')


plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('PCA_explained_variance.jpg')
plt.show()

# %%
