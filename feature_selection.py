# %% 
# Load the package
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# %%
def data_loading(rating_dir, image_dir):
    """
    Read the CSV file and load image data.

    Args:
        rating_dir (str): Directory containing the rating CSV files.
        image_dir (str): Directory containing the images.

    Returns:
        np.ndarray: Array of flattened face images.
        np.ndarray: Array of corresponding ratings.
        tuple: Size of the images.
    """
    Y = []
    X = []

    directory_path = Path(rating_dir)

    print("Reading files...")
    for file in directory_path.rglob('*.csv'):  
        data = pd.read_csv(file)
        
        for img_name in data.ImageName:
            img_path = image_dir + img_name
            
            img=imread(str(img_path))
            img_size = img.shape
            X.append(img.flatten())  # Flatten the image to 1D array
        
        
        Y.extend(data.Rating)
    return np.array(X), np.array(Y), img_size

# %%
def preprocessing(faces, labels):
    """
    Normalize the face images by subtracting the average image, and scale ratings to the range [0, 1].

    Args:
        faces (np.ndarray): Array of flattened face images.
        labels (np.ndarray): Array of ratings.

     Returns:
        np.ndarray: Face images with average image removed.
        np.ndarray: Scaled ratings.
    """

    # Data preprocessing
    print("Subtract the image and Normalize labels...")
    # Compute the average image
    avg = np.mean(faces, axis=0)
    # Subtract the average image from each image
    faces = faces -avg

    # Initialize MinMaxScaler to scale data to range 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape labels into 2D input)
    labels = labels.reshape(-1, 1)
    # Normalize
    normalized_labels = scaler.fit_transform(labels)
    # Flatten it back to 1D 
    normalized_labels = normalized_labels.reshape(faces.shape[0],1)

    return faces, normalized_labels


# %%
def select_features(pca_transformed, labels):
    """
    Use forward selection to select the relevant PCs for the model

    Args:
        pca_transformed (np.ndarray): PCA-transformed face images.
        labels (np.ndarray): Scaled ratings.

    Returns:
        LinearRegression: A linear regression model.
        np.ndarray: Indices of selected PCA components.
    """
    print("Processing Feature Selction...")
    # Create a linear regression model
    lr = LinearRegression()

    # Use sequential feature selector to select 5 features
    n_features_to_select = 5
    sfs_forward = SequentialFeatureSelector(
        lr, n_features_to_select=n_features_to_select, direction='forward'
    )
    # Fit Sequential Feature Selector
    print(f"Selecting {n_features_to_select} features:")
    sfs_forward.fit(pca_transformed , labels)

    # Show the chosen features
    selected_features = sfs_forward.get_support(indices=True)
    # Print the selected features
    print("The selected feature indices are:", selected_features)
    return lr, selected_features


# %%
def visualize_PC(pca_data):
    """
    Visualize images corresponding to the minimum, mean, and maximum projections onto selected PCA components.

    Args:
        pca_data (dict): A dictionary containing the PCA model, average image, image size, transformed data, and selected features.
    """
    print("Visualizing selected PCA components...")
    
    pca = pca_data['pca']
    avg = pca_data['avg']
    img_size = pca_data['img_size']
    pca_transformed = pca_data['pca_transformed']
    selected_features = pca_data['selected_features']

    plt.figure(figsize=[15, 8])
    for i, component in enumerate(selected_features):
        pca_component = pca_transformed[:, component]
        max_idx, min_idx = np.argmax(pca_component), np.argmin(pca_component)
        
        def reconstruct(index):
            pc_space = np.zeros_like(pca_transformed)
            pc_space[:, component] = pca_transformed[:, component]
            return (pca.inverse_transform(pc_space[index]) + avg).reshape(img_size)
        
        plt.subplot(5, 3, 3 * i + 1)
        plt.imshow(reconstruct(min_idx), cmap='gray')
        plt.title(f"Min Projection (PC {component})")
        plt.axis('off')
        
        plt.subplot(5, 3, 3 * i + 2)
        plt.imshow(avg.reshape(img_size), cmap='gray')
        plt.title("Mean Image")
        plt.axis('off')
        
        plt.subplot(5, 3, 3 * i + 3)
        plt.imshow(reconstruct(max_idx), cmap='gray')
        plt.title(f"Max Projection (PC {component})")
        plt.axis('off')
    
    plt.savefig('figures/Selected_PCA_components.jpg')
    plt.show()

#%%
def generate_images(model, ratings):
    """
    Generate images based on provided ratings and a trained model.

    Args:
        model (LinearRegression): Trained linear regression model.
        ratings (list): List of ratings for each feature.

    Returns:
        list: Generated images corresponding to the ratings.
    """
    generated_images = []
    delta, w = model.intercept_, model.coef_
    
    print("Generating images based on provided ratings...")
    for rating in ratings:
        generated_images.append((rating - delta) * w / np.linalg.norm(w) ** 2)
    
    return generated_images


#%%
def visualize_generated(generated, pca_data, ratings_generate):
    """
    Visualize the generated images.

    Args:
        generated_images (list): List of generated images.
        pca_data (dict): Dictionary containing PCA data.
        ratings_generate (list): List of ratings used to generate the images.
    """
    fig, axes = plt.subplots(1, len(ratings_generate), figsize=(20, 5))
    for i, image in enumerate(generated):
        # Calculate the image data
        image_data = (image @ pca_data['pca'].components_[pca_data['selected_features'], :] * 0.05 + pca_data['avg']).reshape(pca_data['img_size'])
         # Display the image in the appropriate subplot
        axes[i].imshow(image_data, cmap='gray')
        axes[i].set_title(f'Rating {ratings_generate[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/Generated_image.jpg')
    plt.show()
    
# %% 
def PCA_transform(ratings_generate):
    """
    Perform PCA, select relevant features, and generate images based on specified ratings.

    Args:
        ratings_generate (list): Ratings to use for generating images.
    """
    
    faces, labels, img_size = data_loading(rating_dir='data/', image_dir="frontalimages_manuallyaligne_greyscale/")
    
    avg = np.mean(faces, axis=0)
    
    normal_faces, normal_labels = preprocessing(faces, labels)

    # Run PCA and transform image to the PCA
    print("Processing PCA....")
    # Fit the PCA
    pca = PCA()
    pca.fit(normal_faces)
    
    # Transform the image
    transformed = pca.transform(normal_faces)

    model, features = select_features(transformed, normal_labels)
    
    pca_data = {
                'pca': pca,
                'avg': avg,
                'img_size': img_size,
                'pca_transformed': transformed,
                'selected_features': features
                }

    visualize_PC(pca_data)
    
    model.fit(transformed[:, features], normal_labels)
    generated = generate_images(model, ratings_generate)
    
    
    visualize_generated(generated, pca_data, ratings_generate)

    

#%% 

if __name__ == "__main__":
    ratings_generate = [-9,-7,-5,-3,-1,0,1,3,5,7,9]
    
    PCA_transform(ratings_generate)

