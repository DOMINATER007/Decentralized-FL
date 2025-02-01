from Models.cnn3 import CNN3
from clients import client_base
import numpy as np

# Define client-specific parameters
client_id = 1
model = CNN3()
dataset_train = "path_to_client0_train.npz"
dataset_test = "path_to_client0_test.npz"
coordinates = (40.7128, -74.0060)  # Example coordinates: New York City
bandwidth = 50  # Mbps

# Initialize client
client0 = client_base(client_id, model, dataset_train, dataset_test, coordinates, bandwidth)