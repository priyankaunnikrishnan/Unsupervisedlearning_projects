# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:15:33 2024

@author: priya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA 
from sklearn.datasets import fetch_openml

# Loading the MNIST dataset from OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist.data.shape)  # Printing the shape of the dataset
X, y = mnist["data"], mnist["target"]  

# Displaying unique digits
unique_digits = np.unique(y)  
num_unique_digits = len(unique_digits)  

# Plotting the unique digits
plt.figure(figsize=(10, 5))  # Set the figure size
for i, digit in enumerate(unique_digits):
    # Get indices of the current digit and randomly select one image
    digit_indices = np.where(y == digit)[0]
    random_index = np.random.choice(digit_indices)
    digit_image = X[random_index].reshape(28, 28)  # Reshape to 28x28 pixels
    
    # Plot each digit in a subplot
    plt.subplot(2, num_unique_digits // 2, i + 1)
    plt.imshow(digit_image, cmap='gray') 
    plt.title(f"Digit {digit}")  
    plt.axis('off')  

plt.tight_layout()  
plt.show()  

# PCA to reduce dimensions to 2 for visualization
pca_mnist = PCA(n_components=2)  # Create a PCA instance with 2 components

# Fit PCA and transform data
X_pca = pca_mnist.fit_transform(X)  # Reduce the dimensionality of the dataset

# Explained variance ratio shows how much information is retained in the components
explained_variance = pca_mnist.explained_variance_ratio_
print("Explained variance ratio for PC1 and PC2: ")
print(explained_variance)

# Plot the projections on PC1 and PC2
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='tab10', s=1)  # Scatter plot for 2D PCA
plt.xlabel("PC1")  # Label for X-axis
plt.ylabel("PC2")  # Label for Y-axis
plt.title("Projections of PC1 and PC2")  # Title for the plot
plt.colorbar(label="Digit class")  # Color bar to show digit classes
plt.show()

# Incremental PCA (useful for large datasets)
n_components = 154  # Seting the number of components to retain
incremental_pca = IncrementalPCA(n_components=n_components, batch_size=200)  # Incremental PCA with batch processing
X_reduced = incremental_pca.fit_transform(X)  
print(f"Shape of reduced dataset: {X_reduced.shape}")

# Reconstructing the data from the reduced dimensions
X_reconstructed = incremental_pca.inverse_transform(X_reduced)  

# Ploting original and reconstructed digits for comparison
num_digits_to_display = 10
plt.figure(figsize=(10, 4))

for i in range(num_digits_to_display):
    # Plot the original digit
    plt.subplot(2, num_digits_to_display, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Original {y[i]}")
    plt.axis('off')

    # Plot the compressed (reconstructed) digit
    plt.subplot(2, num_digits_to_display, i + 1 + num_digits_to_display)
    plt.imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title(f"Compressed {y[i]}")
    plt.axis('off')

plt.tight_layout()  
plt.show()
