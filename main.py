from clients import c1,c2,c3,c4,c5,c6,c7
import random
from geopy.point import Point
from clusteringLogic.dbscan import perform_clustering, plot_clusters
def assign_random(a):
    latitude = random.uniform(10, 20)
    longitude=random.uniform(70,80)
    bandwidth = random.randint(10,200)

    coordinate=Point(latitude,longitude)
    print(f"{coordinate} for client{a} ")
    return coordinate,bandwidth

if __name__ == "__main__":
    clients=[c1.client1,c2.client2,c3.client3,c4.client4,c5.client5,c6.client6,c7.client7]
    a=1
    for i in clients:
        i.coordinates,i.bandwidth=assign_random(a)
        a+=1
    locations = [(c.coordinates[0],c.coordinates[1]) for c in clients]
    
    valid_clusters, cluster_centers, labels, client_coords = perform_clustering(locations,eps=200, min_samples=2)
    plot_clusters(client_coords, labels)
