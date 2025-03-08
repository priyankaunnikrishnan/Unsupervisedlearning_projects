# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:21:48 2024

@author: priya
"""



from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

# Load the Olivetti faces dataset
faces_data = fetch_olivetti_faces()
X, y = faces_data.data, faces_data.target

# Apply StandardScaler to the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled Data:\n", X_scaled)

# Split the dataset into training, validation, and test sets using stratified sampling
split_test = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=64)

# Split into training+validation and test sets using scaled data
for train_val_idx, test_idx in split_test.split(X_scaled, y):
    X_train_val, X_test = X_scaled[train_val_idx], X_scaled[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]

# Split training+validation into training and validation sets
split_val = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=64)
for train_idx, val_idx in split_val.split(X_train_val, y_train_val):
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

# Print the sizes of the splits
print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

# Initialize Random Forest Classifier
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=64)

# Perform K-Fold Cross-Validation on the training data
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=64)
validation_accuracies_rf = []

# Perform K-Fold Cross-Validation
for train_idx, val_idx in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    classifier_rf.fit(X_train_fold, y_train_fold)
    y_pred_fold = classifier_rf.predict(X_val_fold)

    accuracy_fold = accuracy_score(y_val_fold, y_pred_fold)
    validation_accuracies_rf.append(accuracy_fold)

mean_accuracy_rf = np.mean(validation_accuracies_rf)
print(f"Mean Validation Accuracy (Random Forest): {mean_accuracy_rf:.4f}")

# Evaluate the trained model on the Test Set
classifier_rf.fit(X_train, y_train)
y_test_pred = classifier_rf.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_test_pred)
print(f"Test Set Accuracy (Random Forest): {test_accuracy_rf:.4f}")

# Hierarchical Clustering with Euclidean Distance
dist_matrix_euclidean = pdist(X_train, metric='euclidean')
Z_euclidean = linkage(dist_matrix_euclidean, method='centroid')

# Plotting the Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z_euclidean)
plt.title('Dendrogram (Euclidean Distance)')
plt.xlabel('Sample Data Index')
plt.ylabel('Distance')
plt.show()

# Manhattan Distance Clustering
dist_matrix_manhattan = pdist(X_train, metric='minkowski', p=1)
Z_manhattan = linkage(dist_matrix_manhattan, method='centroid')

plt.figure(figsize=(10, 5))
dendrogram(Z_manhattan)
plt.title('Dendrogram (Manhattan Distance with Centroid Linkage)')
plt.xlabel('Sample Data Index')
plt.ylabel('Distance')
plt.show()

# Cosine Distance Clustering
cosine_dist_matrix = pdist(X_train, metric='cosine')
Z_cosine = linkage(cosine_dist_matrix, method='centroid')

plt.figure(figsize=(12, 8))
dendrogram(Z_cosine, truncate_mode='level', p=10, leaf_rotation=90)
plt.title('Dendrogram (Cosine Similarity with Centroid Linkage)')
plt.xlabel('Sample Data Index')
plt.ylabel('Distance')
plt.show()

# Silhouette Scores for Different Distance Metrics
def compute_silhouette_score(X, n_clusters, linkage_method, metric):
    cluster_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage_method
    )
    cluster_labels = cluster_model.fit_predict(X)
    try:
        score = silhouette_score(X, cluster_labels, metric=metric)
    except ValueError:
        score = -1  # Handle poorly formed clusters
    return score

# Compute and Display Silhouette Scores
silhouette_scores = {}
for n_clusters in range(2, 11):
    score_euclidean = compute_silhouette_score(X_train, n_clusters, 'ward', 'euclidean')
    score_manhattan = compute_silhouette_score(X_train, n_clusters, 'average', 'manhattan')
    score_cosine = compute_silhouette_score(X_train, n_clusters, 'average', 'cosine')

    print(f"Number of Clusters: {n_clusters}")
    print(f"  Euclidean Silhouette Score: {score_euclidean:.4f}")
    print(f"  Manhattan Silhouette Score: {score_manhattan:.4f}")
    print(f"  Cosine Silhouette Score: {score_cosine:.4f}")

    silhouette_scores[n_clusters] = {
        'euclidean': score_euclidean,
        'manhattan': score_manhattan,
        'cosine': score_cosine
    }

# Plotting Silhouette Scores
plt.figure(figsize=(10, 6))
for metric in ['euclidean', 'manhattan', 'cosine']:
    scores = [silhouette_scores[k][metric] for k in range(2, 11)]
    plt.plot(range(2, 11), scores, marker='o', label=f'{metric.capitalize()} Distance')

plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Distance Metrics')
plt.legend()
plt.grid(True)
plt.show()

# Find and Apply Optimal Clustering
optimal_clusters = max(silhouette_scores, key=lambda k: silhouette_scores[k]['euclidean'])
print(f"Optimal number of clusters (Euclidean): {optimal_clusters}")

best_cluster_model = AgglomerativeClustering(
    n_clusters=optimal_clusters,
    linkage='ward',
    metric='euclidean'
)
cluster_labels = best_cluster_model.fit_predict(X_train)
print(f"Clusters assigned to the training data: {np.unique(cluster_labels)}")


from sklearn.manifold import MDS
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Step 4: Calculate Distance Matrices
dist_matrix_euclidean = squareform(pdist(X_train_val, metric='euclidean'))
dist_matrix_manhattan = squareform(pdist(X_train_val, metric='minkowski', p=1))
dist_matrix_cosine = squareform(pdist(X_train_val, metric='cosine'))

# Step 5: Define a function to perform K-Fold CV on transformed data
def train_with_distance_matrix(dist_matrix, y, n_splits=5):
    # Initialize MDS to transform the distance matrix into a feature space
    mds = MDS(n_components=10, dissimilarity='precomputed', random_state=64)
    X_transformed = mds.fit_transform(dist_matrix)

    # Set up K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=64)
    classifier = RandomForestClassifier(n_estimators=100, random_state=64)
    accuracies = []

    # Perform K-Fold CV
    for train_idx, val_idx in kf.split(X_transformed, y):
        X_train_fold, X_val_fold = X_transformed[train_idx], X_transformed[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Train and evaluate the model
        classifier.fit(X_train_fold, y_train_fold)
        y_pred_fold = classifier.predict(X_val_fold)
        accuracies.append(accuracy_score(y_val_fold, y_pred_fold))

    return np.mean(accuracies)

# Step 6: Perform K-Fold CV on each distance matrix
mean_accuracy_euclidean = train_with_distance_matrix(dist_matrix_euclidean, y_train_val)
print(f"Mean Validation Accuracy (Euclidean Distance): {mean_accuracy_euclidean:.4f}")

mean_accuracy_manhattan = train_with_distance_matrix(dist_matrix_manhattan, y_train_val)
print(f"Mean Validation Accuracy (Manhattan Distance): {mean_accuracy_manhattan:.4f}")

mean_accuracy_cosine = train_with_distance_matrix(dist_matrix_cosine, y_train_val)
print(f"Mean Validation Accuracy (Cosine Distance): {mean_accuracy_cosine:.4f}")



      
      
