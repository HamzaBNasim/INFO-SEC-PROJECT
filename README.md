# INFO-SEC-PROJECT
we will be creating a repository for the project
from sklearn.neighbors import NearestNeighbors
import numpy as np


def dbscan(dataset, epsilon, min_points):
    """
    Perform DBSCAN clustering on a dataset.

    :param dataset: The dataset as a numpy array.
    :param epsilon: The maximum distance between two samples for them to be considered as neighbors.
    :param min_points: The minimum number of samples in a neighborhood for a point to be considered as a core point.
    :return: A list of cluster labels for each data point. Outliers are labeled as -1.
    """
    # Step 1: Initialize variables
    cluster_label = 0
    visited = set()
    clusters = np.zeros(len(dataset))  # 0 represents unvisited points

    # Step 2: Compute neighborhoods
    neighbors = NearestNeighbors(n_neighbors=min_points).fit(dataset)
    distances, indices = neighbors.kneighbors(dataset)

    # Step 3: Start clustering
    for i in range(len(dataset)):
        if i in visited:
            continue

        visited.add(i)
        neighbors_i = indices[i].tolist()

        if len(neighbors_i) >= min_points:
            cluster_label += 1
            expand_cluster(dataset, i, neighbors_i, cluster_label, epsilon, min_points, visited, clusters)
        else:
            clusters[i] = -1  # Label as outlier

    return clusters


def expand_cluster(dataset, point_index, neighbors, cluster_label, epsilon, min_points, visited, clusters):
    """
    Expand the cluster starting from a core point.

    :param dataset: The dataset as a numpy array.
    :param point_index: Index of the core point to start expanding from.
    :param neighbors: List of indices of the neighbors of the core point.
    :param cluster_label: Cluster label for the core point.
    :param epsilon: The maximum distance between two samples for them to be considered as neighbors.
    :param min_points: The minimum number of samples in a neighborhood for a point to be considered as a core point.
    :param visited: Set of visited indices.
    :param clusters: Array to store cluster labels for each data point.
    """
    clusters[point_index] = cluster_label

    i = 0
    while i < len(neighbors):
        neighbor_index = neighbors[i]
        if neighbor_index not in visited:
            visited.add(neighbor_index)
            new_neighbors = indices[neighbor_index].tolist()

            if len(new_neighbors) >= min_points:
                neighbors.extend(new_neighbors)  # Add new neighbors to the list
            if clusters[neighbor_index] == 0 or clusters[neighbor_index] == -1:
                clusters[neighbor_index] = cluster_label

        i += 1


# Example usage
dataset = np.array([[1, 1], [1.5, 2], [2, 1], [10, 8], [10, 10], [10, 11]])
epsilon = 1.5
min_points = 2

cluster_labels = dbscan(dataset, epsilon, min_points)
print(cluster_labels)

