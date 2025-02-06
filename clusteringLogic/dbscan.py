import random
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Generate random client coordinates for testing
def generate_random_coordinates(num_clients=20, lat_range=(10, 30), lon_range=(70, 90)):
    return {
        f"Client{i+1}": (random.uniform(*lat_range), random.uniform(*lon_range))
        for i in range(num_clients)
    }

# Generate 20 random client coordinates
client_coordinates = generate_random_coordinates(num_clients=20)

def haversine(coord1, coord2):
    # HAVERSINE DISTANCE
    return geodesic(coord1, coord2).km

def perform_clustering(client_coords,eps=200, min_samples=2):
    # Convert client coordinates to numpy array
    client_coords = np.array(client_coords)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=lambda x, y: haversine(x, y))
    labels = dbscan.fit_predict(client_coords)

    # Assign coordinates to clusters
    clusters = {}
    for idx, label in enumerate(labels):
        if label != -1:  # Ignore noise points (label = -1)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx+1)
    print("\n==============777==================\n")
    print(clusters)
    print("\n================777================\n")
    # Filter clusters based on size constraints (at least 2 points per cluster)
    valid_clusters = {i: cluster for i, cluster in clusters.items() if len(cluster) >= 2}

    # Compute cluster centers for valid clusters
    cluster_centers = {
        i: np.mean([client_coords[key-1] for key in cluster], axis=0).tolist()
        for i, cluster in valid_clusters.items()
    }

    return valid_clusters, cluster_centers, labels, client_coords

def plot_clusters(client_coords, labels):
    # Visualize clusters
    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = "black"  # Noise points
        cluster_points = client_coords[np.array(labels) == label]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c=[color], label=f"Cluster {label}")

    plt.title("DBSCAN Clustering")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid()
    plt.show()

# Perform clustering and plot results
# valid_clusters, cluster_centers, labels, client_coords = perform_clustering(eps=200, min_samples=2)
# plot_clusters(client_coords, labels)