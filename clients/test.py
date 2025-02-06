import json
import socket
import threading
import torch
import torch.nn as nn

class ClientBase:
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
        self.server_socket.bind(('localhost', 8080 + client_id))
        print(f"Client {client_id} bind completed on port {8080 + client_id}")
        self.server_address = self.server_socket.getsockname()
        self.server_port = self.server_socket.getsockname()[1]
        self.server_socket.listen(10)
        self.data_of_others = {}
        self.server_thread = threading.Thread(target=self.receive_data_from_other_clients, daemon=True)
        self.server_thread.start()

    def json_encode1(self):
        data = {
            'location': self.coordinates,
            'id': self.client_id
        }
        return json.dumps(data)

    def json_encode2(self):
        data = {
            "model_architecture": str(self.model.__class__.__name__),
            "coordinates": self.coordinates,
            "bandwidth": self.bandwidth,
            "accuracy_history_list": self.accuracy_history_list,
            "client_id": self.client_id,
        }
        return json.dumps(data)

    def broadcast(self, peers):
        message = self.json_encode1()
        for addr, port in peers:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                client_socket.connect((addr, port))
                client_socket.sendall(message.encode('utf-8'))
            except Exception as e:
                print(f"Failed to connect to {addr}:{port} - {e}")
            finally:
                client_socket.close()

    def receive_data_from_other_clients(self):
        while True:
            client_socket, client_addr = self.server_socket.accept()
            try:
                data = client_socket.recv(1024).decode('utf-8')
                print(f"Client {self.client_id} received data {data} from {client_addr}")
                self.data_of_others[client_addr] = data
            except Exception as e:
                print(f"Exception occurred {e}")
            finally:
                client_socket.close()

    def send_direct_message(self, message, addr, port):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((addr, port))
            client_socket.sendall(message.encode('utf-8'))
        except Exception as e:
            print(f"Failed to connect to {addr}:{port} - {e}")
        finally:
            client_socket.close()

    def run(self, peers):
        print(f"Client {self.client_id} is running...")
        while True:
            action = input(f"Client {self.client_id}: Enter 'broadcast' to broadcast, 'send' to send a direct message, or 'exit' to quit: ")
            if action == 'broadcast':
                self.broadcast(peers)
            elif action == 'send':
                target_id = int(input("Enter target client ID: "))
                target_port = 8080 + target_id
                message = input("Enter message: ")
                self.send_direct_message(message, 'localhost', target_port)
            elif action == 'exit':
                break
        self.server_socket.close()

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    print(client_id)
    # Dummy model, dataset, and other parameters
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 1)

    model = DummyModel()
    dataset_train = [1, 2, 3]
    dataset_test = [4, 5, 6]
    coordinates = (1.0, 2.0)
    bandwidth = 100

    num_clients = 3  # Total number of clients
    peers = [(f'localhost', 8080 + i) for i in range(num_clients) if i != client_id]

    client = ClientBase(client_id, model, dataset_train, dataset_test, coordinates, bandwidth)
    client.run(peers)