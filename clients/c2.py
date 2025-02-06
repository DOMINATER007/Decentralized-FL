import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.cnn5 import CNN5
from clients import client_base
from geopy.point import Point

import random
# Define client-specific parameters
client_id = 2
model = CNN5()
dataset_train = "./DataDistribution/client_datasets/client_2_train.npz"
dataset_test = "./DataDistribution/client_datasets/client_2_test.npz"
latitude = random.uniform(-90, 90)
longitude=random.uniform(-180,180)
bandwidth = random.randint(10,200)

coordinate=Point(latitude,longitude)


# Initialize client
client2 = client_base(client_id, model, dataset_train, dataset_test, coordinate, bandwidth)