# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:53:00 2024

@author: priya
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X,color = make_swiss_roll(n_samples=1000,noise =0.1,random_state=64, hole=False)
y = (color > 9).astype(int)
# Plot the generated dataset
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c= color ,cmap='rainbow')
ax.view_init(10,-70)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.title('Swiss Roll Dataset')
plt.show()

kernels = ['linear', 'rbf','sigmoid']
kpca_results = {}

for kernel in kernels:
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=0.1)
    X_kpca = kpca.fit_transform(X)
    kpca_results[kernel] = X_kpca

    # Plotting the kPCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=X_kpca[:, 1], cmap='autumn', s=20)
    plt.title(f'Kernel PCA with {kernel} kernel')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Color Gradient')
    plt.show()
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)
pipeline = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression())
])

param_grid = {
    'kpca__kernel': ['rbf'],  
    'kpca__gamma': np.logspace(-2, 1, 10)  # Testing gamma values
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Print the best classification accuracy on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Best classification accuracy on the test set: {test_accuracy:.4f}")

# Plot the GridSearchCV results
results = grid_search.cv_results_

# Extract gamma values and corresponding mean test scores
gamma_values = param_grid['kpca__gamma']
mean_test_scores = results['mean_test_score']

plt.figure(figsize=(8, 6))
plt.plot(gamma_values, mean_test_scores, marker='o', linestyle='-', color='b')
plt.xscale('log')  # Gamma is on a log scale
plt.xlabel('Gamma (log scale)')
plt.ylabel('Mean cross-validation accuracy')
plt.title('GridSearchCV Results: Gamma vs Accuracy')
plt.grid(True)
plt.show()