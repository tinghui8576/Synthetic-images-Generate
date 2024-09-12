# %%
import cv2
import numpy as np
import pandas as pd
from matplotlib.image import imread
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
# Read the CSV file and save it to X and Y
Y = []
X = []
directory_path = Path('data/')


# Set the target size for resizing the images (e.g., 128x128)

print("Reading files...")
for file in directory_path.rglob('*.csv'):  
    data = pd.read_csv(file)
    
    for img_name in data.ImageName:
        img_path = "frontalimages_manuallyaligne_greyscale/" + img_name
        
        img=imread(str(img_path))
        
        
        
        X.append(img.flatten())  # Flatten the image to 1D array
    
    # Extend the labels
    Y.extend(data.Rating)
    # Convert list of flattened images to a numpy array
faces = np.array(X)
labels = np.array(Y)


# %%
print("Normalize the image and labels...")
# Compute the average image
avg = np.mean(faces, axis=0)

# Subtract the average image from each image
faces = faces -avg
# %%
# Initialize MinMaxScaler to scale data to range 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Reshape labels if it's a 1D array (scikit-learn expects 2D input)
labels = labels.reshape(-1, 1)

# Fit the scaler to your data and transform it
normalized_labels = scaler.fit_transform(labels)

# If you want to flatten it back to 1D (in case of a single feature):
normalized_labels = normalized_labels.reshape(faces.shape[0],1)


# %% 
print("Processing PCA....")
pca = PCA()
pca.fit(faces)

components = pca.components_
# Visually present the first few PCA components as images
pca_transformed = pca.transform(faces)

# %%
print("Processing Feature Selction...")
lr = LinearRegression()

# Sequential Feature Selector
n_features_to_select = 5
sfs_forward = SequentialFeatureSelector(
    lr, n_features_to_select=n_features_to_select, direction='forward'
)


# Fit Sequential Feature Selector
print(f"Selecting {n_features_to_select} features:")
with tqdm(total=n_features_to_select) as pbar:
    sfs_forward.fit(pca_transformed , normalized_labels)
    for _ in range(n_features_to_select):
        pbar.update(1)  # Update progress for each feature selected

# %%
selected_features = sfs_forward.get_support(indices=True)

# Print the selected features
print("The selected feature indices are:", selected_features)

# Optionally, you can retrieve the actual PCA components associated with the selected features
print("The selected PCA components are:")
for idx in selected_features:
    print(idx, pca.components_[idx])
# %%a
plt.figure(figsize=[15,8])
for i, component_index in enumerate(selected_features):

    # Select the PCA component to analyze 
    pca_component = pca_transformed[:, component_index]
    # Find the images with the maximum and minimum projection scores for the component
    max_score_index = np.argmax(pca_component)
    min_score_index = np.argmin(pca_component)

    # Reconstruct images 
    def reconstruct(index, n_components=1):
        # Keep only the first n_components, set the rest to 0
        pca_keep = np.zeros_like(pca_transformed)
        pca_keep[:, n_components] = pca_transformed[:, n_components]
        return (pca.inverse_transform(pca_keep[index])+avg).reshape(img.shape)

    max_reconstructed = reconstruct(max_score_index, n_components=component_index)
    min_reconstructed = reconstruct(min_score_index, n_components=component_index)

    # Visualize the min, mean, and max images
    # Min Image
    plt.subplot(5, 3, 3*i + 1)  # 3 rows, 3 columns, fill in order
    plt.imshow(min_reconstructed, cmap='gray')
    plt.title(f"Min Comp {component_index}")
    plt.axis('off')
    
    # Mean Image
    plt.subplot(5, 3, 3*i + 2)
    plt.imshow(avg.reshape(img.shape), cmap='gray')
    plt.title(f"Mean Image")
    plt.axis('off')

    # Max Image
    plt.subplot(5, 3, 3*i + 3)
    plt.imshow(max_reconstructed, cmap='gray')
    plt.title(f"Max Comp {component_index}")
    plt.axis('off')
plt.savefig('Selected_PCA_component.jpg')
plt.show()
# %%
