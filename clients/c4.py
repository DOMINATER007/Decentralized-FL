import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.cnn3 import CNN3
from clients.client_base import client_base
from geopy.point import Point

import random
# Define client-specific parameters
client_id = 4
model = CNN3()
dataset_train = r"E:\MAJORPROJECT\Decentralized-FL\DataDistribution\client_datasets\client_4_train.npz"
dataset_test = r"E:\MAJORPROJECT\Decentralized-FL\DataDistribution\client_datasets\client_4_test.npz"
latitude = random.uniform(-90, 90)
longitude=random.uniform(-180,180)
bandwidth = random.randint(10,200)

coordinate=Point(latitude,longitude)


# Initialize client
client4 = client_base(client_id, model, dataset_train, dataset_test, coordinate, bandwidth)