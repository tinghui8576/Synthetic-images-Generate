# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
# Read the CSV file and save it to X and Y
Y = []
directory_path = Path('data/')


for file in directory_path.rglob('*.csv'):  
    data = pd.read_csv(file)
    
    Y.extend(data.Rating)
    # Convert list of flattened images to a numpy array

labels = np.array(Y)

# %%
labels
# %%
# Plot the distribution of labels using a histogram
plt.figure(figsize=(8, 6))
sns.countplot(x=labels, palette="viridis")
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()



# %%
Y