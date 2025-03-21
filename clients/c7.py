
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.cnn5 import CNN5
from clients.client_base import client_base
from geopy.point import Point
import numpy as np
import random
# Define client-specific parameters
client_id = 7
model = CNN5()
dataset_train = r"E:\MAJORPROJECT\Decentralized-FL\DataDistribution\client_datasets\client_7_train.npz"
dataset_test = r"E:\MAJORPROJECT\Decentralized-FL\DataDistribution\client_datasets\client_7_test.npz"
latitude = random.uniform(-90, 90)
longitude=random.uniform(-180,180)
bandwidth = random.randint(10,200)

coordinate=Point(latitude,longitude)


# Initialize client
client7 = client_base(client_id, model, dataset_train, dataset_test, coordinate, bandwidth)

train_data=np.load(dataset_train)
test_data=np.load(dataset_test)

print(train_data)