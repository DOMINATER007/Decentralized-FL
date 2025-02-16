import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.cnn7 import CNN7
from clients.client_base import client_base
from geopy.point import Point

import random
# Define client-specific parameters
client_id = 3
model = CNN7()
dataset_train = r"/mnt/c/Users/karra/Desktop/Federated_Learning/DataDistribution/client_datasets/client_3_train.npz"
dataset_test = r"/mnt/c/Users/karra/Desktop/Federated_Learning/DataDistribution/client_datasets/client_3_test.npz"
latitude = random.uniform(-90, 90)
longitude=random.uniform(-180,180)
bandwidth = random.randint(10,200)

coordinate=Point(latitude,longitude)


# Initialize client
client3 = client_base(client_id, model, dataset_train, dataset_test, coordinate, bandwidth)