#from clients import c1,c2,c3,c4,c5,c6,c7
import random
import json
import threading
from FKD import feature_based_kd
from geopy.point import Point
from clusteringLogic.dbscan import perform_clustering, plot_clusters
from election import election_helper
import matplotlib.pyplot as plt
import numpy as np
from DataDistribution.data_distributor import data_prep
import os
import random
from geopy.point import Point
from Models.cnn3 import CNN3  # Assuming CNN3 is one of the models
from Models.cnn5 import CNN5 
from Models.cnn7 import CNN7 # Add more models as needed
from clients.client_base import client_base

# Define available models
available_models = [CNN3, CNN5,CNN7]  # Extend this list with other models

def assign_random(a):
    latitude = random.uniform(10, 20)
    longitude=random.uniform(70,80)
    bandwidth = random.randint(10,200)

    coordinate=Point(latitude,longitude)
    print(f"{coordinate} for client{a} ")
    return coordinate,bandwidth

def create_clients(n, data_path):
    clients = []
    for i in range(1, n + 1):
        print(f"\nCLIENT {i} has MODEL {available_models[(i-1)%3]}\n")
        model = (available_models[(i-1)%3])()
        dataset_train = os.path.join(data_path, f"client_{i}_train.npz")
        dataset_test = os.path.join(data_path, f"client_{i}_test.npz")
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)
        bandwidth = random.randint(10, 200)
        coordinate = Point(latitude, longitude)
        p_rate=random.uniform(0.3, 0.8)
        print(f"Client {i} is at {coordinate} with bandwidth {bandwidth}.")
        client = client_base(i, model, dataset_train, dataset_test, coordinate, bandwidth,p_rate)
        clients.append(client)
    
    return clients


if __name__ == "__main__":
    #clients=[c1.client1,c2.client2,c3.client3,c4.client4,c5.client5,c6.client6,c7.client7]
    data_path = r"E:\MAJORPROJECT\Decentralized-FL\DataDistribution\client_datasets"
    alpha = 0.7  # Controls the class imbalance among clients
    save_dir = r"DataDistribution\client_datasets"
    n_clients = 10  # Change as needed
    print("Processing data for all clients...")
    data_prep(n_clients, alpha, save_dir)

    clients = create_clients(n_clients, data_path)
    accuracy_history = {client.client_id: [] for client in clients}
    accuracy_per_round = []
    a=1
    for i in clients:
        i.coordinates,i.bandwidth=assign_random(a)
        a+=1

    for i in range(1,3):
        round_accuracy = {}
        for client in clients:
            round_accuracy[client.client_id] = {'before_KD': 0.0, 'after_KD': 0.0}
        print(f"\n\n --------------------Round :{i}-------------- \n\n")
        if i==1:
            for client in clients:
                print(f"\nRUNNING client {client.client_id}\n")
                accuracy,_,_=client.training_phase()
                #print(_)   PRINTING PENULTIMATE OUTPUTS
                accuracy_history[client.client_id].append(accuracy)
                round_accuracy[client.client_id] = {'before_KD': accuracy, 'after_KD': accuracy}
                print(f"Model Accuracy of client{client.client_id}: {accuracy:.4f}")
        
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
            
            for c in cluster:
                clients[c-1].cluster_id=cluster_id
                
            willing_clients=[]
            for c in cluster:
                chosen_val=random.uniform(0,1)
                print(f"client {c} has chosen value {chosen_val} where the fixed value is {clients[c-1].fixed_participation_rate}\n")
                if clients[c-1].fixed_participation_rate<chosen_val:
                    willing_clients.append(c)
            print(f"Cluster {cluster_id} has {len(willing_clients)} willing clients\n")
            if len(willing_clients)<=1:
                continue
            
            for c in willing_clients:
                peers=[(f'localhost',clients[i-1].server_port) for i in willing_clients if i!=c]
                json_data=clients[c-1].json_encode2()
                clients[c-1].braoadCast(peers)
                
            data={}     
            for c in willing_clients:
                #print(clients[c-1].data_of_others)
                for key,val in clients[c-1].data_of_others.items():
                    #print(f"***key : {key}\n")
                    for k,v in val.items():
                        data[key]=v
            #print("data:{data}\n")
                       
            leader=election_helper(willing_clients,data)
            clients[leader-1].freq_elected_as_leader+=1
            print(f"Leader of cluster {cluster_id} is {leader}")
            
            
            peers=[(f'localhost',clients[i-1].server_port) for i in willing_clients if i!=leader]
            clients[leader-1].broadcast_weights(peers)
            if i>1:
                for c in cluster:
                    round_accuracy[c]['before_KD'] = accuracy_per_round[i-2][c]['after_KD']
                    round_accuracy[c]['after_KD'] = accuracy_per_round[i-2][c]['after_KD']
            for c in willing_clients:
                if c!=leader:
                    ac=feature_based_kd(clients[leader-1],clients[c-1])
                    print(f"\n***Distillation Accuracy of client {c} : {ac}***********\n")
                    round_accuracy[c]['after_KD'] = ac
            for c in cluster:
                ac=clients[c-1].testing_phase()
                if i>1:
                    accuracy_history[c].append(ac)
                clients[c-1].accuracy_history_list.append(ac)
            print("\n\n")
        accuracy_per_round.append(round_accuracy)    
                    

            # EACH CLUSTER HAVE PENULTIMATE WEIGHTS SHARED AND STORED IN THE FORM OF DICTIONARY OF DICTIONARIES
            
            
            
    # print(f"\nACCURACIES : {accuracy_history}\n")     
    # print(f"\nACCURACIES per round : {accuracy_per_round}\n") 
    # # Plot accuracy of each client over rounds
    # plt.figure(figsize=(10, 5))
    # for client_id, acc_list in accuracy_history.items():
    #     plt.plot(range(2), acc_list, label=f'Client {client_id}')
    # plt.xlabel("Rounds")
    # plt.ylabel("Accuracy")
    # plt.title("Client Accuracy Across Rounds")
    # plt.legend()
    # import time
    # plt.savefig("client_accuracy_over_rounds{}.png".format(time.time))
    # plt.show()     
    
    
    # # Plot accuracy trend
    # plt.figure(figsize=(10, 6))
    # for client in clients:
    #     before_KD_acc = [round_acc[client.client_id]['before_KD'] for round_acc in accuracy_per_round]
    #     after_KD_acc = [round_acc[client.client_id]['after_KD'] for round_acc in accuracy_per_round]
    #     plt.plot(range(2), before_KD_acc, linestyle='dashed', label=f'Client {client_id} Before KD')
    #     plt.plot(range(2), after_KD_acc, label=f'Client {client_id} After KD')

    # plt.xlabel('Rounds')
    # plt.ylabel('Accuracy')
    # plt.title('Client Accuracy Before and After KD Over Rounds')
    # plt.legend()
    # plt.show()      
