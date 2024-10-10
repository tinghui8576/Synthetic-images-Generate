# %%
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
# Read the CSV file and save it to X and Y
Y = []
directory_path = Path('data/face_rating')


for i, file in enumerate(directory_path.rglob('*.csv')):  
    data = pd.read_csv(file)
    
    Y.extend(data.Rating)
    print(f"For tester {i+1}, he/she use a range from {max(data.Rating)} to {min(data.Rating)} for rating")
    # Convert list of flattened images to a numpy array

labels = np.array(Y)

# %%
# Plot the distribution of labels using a histogram
plt.figure(figsize=(8, 6))
sns.countplot(x=labels, palette="viridis")
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig("figures/distribution.jpg")
plt.show()


# %%
# Initialize MinMaxScaler to scale data to range 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Reshape labels if it's a 1D array (scikit-learn expects 2D input)
labels = labels.reshape(-1, 1)

# Fit the scaler to your data and transform it
normalized_labels = scaler.fit_transform(labels)

# If you want to flatten it back to 1D (in case of a single feature):
normalized_labels = normalized_labels.flatten()

# Print normalized labels
print("Normalized Labels:", normalized_labels)
# %%
