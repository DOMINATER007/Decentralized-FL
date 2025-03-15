from clients import c1,c2,c3,c4,c5,c6,c7
import random
import json
import threading
from FKD import feature_based_kd
from geopy.point import Point
from clusteringLogic.dbscan import perform_clustering, plot_clusters
from election import election_helper
def assign_random(a):
    latitude = random.uniform(10, 20)
    longitude=random.uniform(70,80)
    bandwidth = random.randint(10,200)

    coordinate=Point(latitude,longitude)
    print(f"{coordinate} for client{a} ")
    return coordinate,bandwidth

if __name__ == "__main__":
    clients=[c1.client1,c2.client2,c3.client3,c4.client4,c5.client5,c6.client6,c7.client7]
    for client in clients:
        print(f"\nRUNNING client {client.client_id}\n")
        accuracy,_,_=client.training_phase()
        #print(_)   PRINTING PENULTIMATE OUTPUTS
        print(f"Model Accuracy of client{client.client_id}: {accuracy:.4f}")
    a=1
    for i in clients:
        i.coordinates,i.bandwidth=assign_random(a)
        a+=1
    locations = [(c.coordinates[0],c.coordinates[1]) for c in clients]
    
    valid_clusters, cluster_centers, labels, client_coords = perform_clustering(locations,eps=200, min_samples=2)
    print(cluster_centers)
    # print("\n================================\n")
    # print(labels)
    # print("\n================================\n")
    # print(client_coords)
    # print("\n================================\n")
    # print(valid_clusters)
    #plot_clusters(client_coords, labels)
    for cluster_id,cluster in valid_clusters.items():
        print(f"Cluster {cluster_id} has {len(cluster)} clients")
        for c in cluster:
            print(f"Client {c}")
        print("\n")
        data={}  
        for c in cluster:
            clients[c-1].cluster_id=cluster_id
        for c in cluster:
            peers=[(f'localhost',clients[i-1].server_port) for i in cluster if i!=c]
            json_data=clients[c-1].json_encode2()
            clients[c-1].braoadCast(peers)
            
        for c in cluster:
            
            print(clients[c-1].data_of_others)
            for key,val in clients[c-1].data_of_others.items():
                print(f"***key : {key}\n")
                for k,v in val.items():
                    data[key]=v
        print("data:{data}\n")            
        leader=election_helper(cluster,data)
        print(f"Leader of cluster {cluster_id} is {leader}")
        
        
        peers=[(f'localhost',clients[i-1].server_port) for i in cluster if i!=leader]
        clients[leader-1].broadcast_weights(peers)
        for c in cluster:
            if c!=leader:
                ac=feature_based_kd(clients[leader-1],clients[c-1])
                

        # EACH CLUSTER HAVE PENULTIMATE WEIGHTS SHARED AND STORED IN THE FORM OF DICTIONARY OF DICTIONARIES
        
        
        
        print("\n\n")
        
        
