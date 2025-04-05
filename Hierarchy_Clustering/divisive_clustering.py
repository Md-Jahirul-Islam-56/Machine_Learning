import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def divisive_clustering(X, max_clusters=5):
    """
    Perform Divisive Hierarchical Clustering using K-Means (k=2) recursively.
    Stop when the number of clusters reaches max_clusters.
    """
    clusters = {0: X}  # Start with one cluster
    cluster_labels = np.zeros(X.shape[0], dtype=int)
    next_label = 1
    
    while len(clusters) < max_clusters:
        # Find the cluster with max variance to split
        cluster_to_split = max(clusters, key=lambda k: np.var(clusters[k]))
        data_to_split = clusters[cluster_to_split]

        # Apply KMeans to split into 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_to_split)

        # Create new clusters
        new_clusters = {
            next_label: data_to_split[labels == 0],
            next_label + 1: data_to_split[labels == 1]
        }

        # Remove old cluster and add new clusters
        del clusters[cluster_to_split]
        clusters.update(new_clusters)

        # Update cluster labels
        for key, data in new_clusters.items():
            mask = np.isin(X, data).all(axis=1)
            cluster_labels[mask] = key

        next_label += 2  # Increment label count
    
    return cluster_labels

# Generate synthetic data
np.random.seed(42)
X = np.vstack([np.random.rand(10, 2) + i for i in range(3)])  # 3 cluster dataset

# Perform divisive clustering
cluster_labels = divisive_clustering(X, max_clusters=3)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k')
plt.title("Divisive Clustering Results")
plt.show()
