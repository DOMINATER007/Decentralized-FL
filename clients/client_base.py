import socket
import threading
import pickle
import torch
import torch.nn as nn

class client_base:
    def __init__(self, client_id, model, dataset_train, dataset_test, coordinates, bandwidth):
        self.client_id = client_id
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.coordinates = coordinates
        self.bandwidth = bandwidth
        self.cluster_id = None
        self.accuracy_history_list = []
        self.freq_elected_as_leader = 0
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', 8080+client_id))
        self.server_address=self.server_socket.getsockname()
        self.server_port = self.server_socket.getsockname()[1]
        self.server_socket.listen()
        self.server_thread = threading.Thread(target=self.server)
        self.server_thread.start()

    def send_data_to_other_clients(self, other_clients):
        for client in other_clients:
            if client.client_id != self.client_id:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(("localhost", 8080 + client.client_id))  # Example port mapping
                    data = {
                        "model_architecture": str(self.model.__class__.__name__),
                        "coordinates": self.coordinates,
                        "bandwidth": self.bandwidth,
                        "accuracy_history_list": self.accuracy_history_list,
                        "client_id": self.client_id,
                    }
                    s.sendall(str(data).encode())
                    s.close()
                except Exception as e:
                    print(f"Failed to send data to client {client.client_id}: {e}")


    def receive_data_from_other_clients(self):
        
        while True:
            conn, addr = self.server_socket.accept()
            data = conn.recv(1024).decode()
            print(f"Received data from {addr}: {data}")
            conn.close()

    def get_penultimate_layer_outputs(self, x):
        layers = list(self.model.children())
        penultimate_layer = layers[-2]
        with torch.no_grad():
            outputs = penultimate_layer(x)
        return outputs