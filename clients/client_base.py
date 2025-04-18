import json
import socket
import threading
import pickle
import torch
import numpy as np
import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset


class client_base:
    def __init__(self, client_id, model, dataset_train, dataset_test, coordinates, bandwidth,p_rate):
        self.client_id = client_id
        self.model = model
        self.model_info={}
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.coordinates = coordinates
        self.bandwidth = bandwidth
        self.cluster_id = None
        self.accuracy_history_list = []
        self.freq_elected_as_leader = 0
        self.fixed_participation_rate = p_rate
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', 8080+self.client_id))
        print("bind completed")
        self.server_address=self.server_socket.getsockname()[0]
        print("server_address",self.server_address)
        self.server_port = self.server_socket.getsockname()[1]
        self.server_socket.listen(10)
        self.server_thread = threading.Thread(target=self.recvHelper, daemon=True)
        print(self.server_port)
        self.server_thread.start()
        #self.server_thread.join()
        self.data_of_others={}
        self.penultimate_outputs=[]

    def json_encode1(self):
        data = {
            'location': (self.coordinates.latitude,self.coordinates.longitude),
            'id': self.client_id
        }
        return json.dumps(data)
    
    def json_encode2(self):
        data = dict()
        data['coordinates']=(self.coordinates.latitude,self.coordinates.longitude)
        data['bandwidth']=self.bandwidth
        data['accuracy_history_list']=self.accuracy_history_list
        data['client_id']=self.client_id
        data['History']=self.freq_elected_as_leader
        data['Model_info']=self.model_info
        
        obj=json.dumps(data)
        return obj   
    def json_encode3(self):
        data = dict()
        data['penultimate_outputs'] = self.penultimate_outputs.tolist()
        return json.dumps(data)
    def broadcast_weights(self,peers):
        for addr,port in peers:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #print(addr,port)
            client_socket.connect((addr,port))
            client_socket.send(self.json_encode3().encode('utf-8'))
    # def braoadCast(self,peers):
    #     for addr,port in peers:
    #         client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         print(addr,port)
    #         client_socket.connect((addr,port))
    #         client_socket.send(self.json_encode2().encode('utf-8'))
    def braoadCast(self, peers):
        self.received_acks = 0  # Track received acknowledgments
        self.expected_acks = len(peers)  # Number of peers to acknowledge
        self.ack_event = threading.Event()
        self.ack_lock = threading.Lock()

        def send_data(addr, port):
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((addr, port))
                client_socket.send(self.json_encode2().encode('utf-8'))
                
                # Wait for acknowledgment
                ack = client_socket.recv(1024).decode('utf-8')
                if ack == "ACK":
                    with self.ack_lock:
                        self.received_acks += 1
                        if self.received_acks == self.expected_acks:
                            self.ack_event.set()  # Notify that all acks are received
                
                client_socket.close()
            except Exception as e:
                print(f"Broadcast error to {addr}:{port} - {e}")

            # Start broadcasting in threads
        threads = []
        for addr, port in peers:
            t = threading.Thread(target=send_data, args=(addr, port))
            threads.append(t)
            t.start()
        
        # Wait for all acknowledgments before proceeding
        self.ack_event.wait()
          
    def recvHelper(self):
        while True:
            client_soc,client_addr=self.server_socket.accept()
            try:
                data=client_soc.recv(1024).decode('utf-8')
                data=json.loads(data)

              #  print(f"Received Data {data} from {client_addr}")
                cID=data["client_id"]
           
                if cID not in self.data_of_others:
            
                    self.data_of_others[cID]={}
         
                self.data_of_others[cID][len(self.data_of_others[cID])]=data
       
                #print(f"Received Data {data} from {client_addr}")
                client_id=data["client_id"]
                print(f"\nClient ID {client_id}\n")
                if client_id not in self.data_of_others:
                    self.data_of_others[client_id]={}
                self.data_of_others[client_id][len(self.data_of_others[client_id])]=data
                #print(self.data_of_others)
            #    print("NOOOOOOOO5")
                client_soc.send("ACK".encode('utf-8'))
            except Exception as e:
                print(f"Excepton Occured {e}")
            finally:
                client_soc.close()  
    def receive_end(self):
        listening_thread=threading.Thread(target=self.recvHelper)
        listening_thread.start()
    def training_phase(self):
        accuracy=self.model.train_model(self.dataset_train)
        self.penultimate_outputs=accuracy["penultimate_outputs"]
        self.model_info=accuracy["model_info"]
        self.accuracy_history_list.append(accuracy["training_accuracy"])
        return accuracy["training_accuracy"],accuracy["penultimate_outputs"],accuracy["model_info"]
    def testing_phase(self):
        accuracy=self.model.test_model(self.dataset_test)
        print(f"Model Accuracy: {accuracy:.4f}")
        return accuracy    

    # def load_data(self):
    #     # Load training data
    #     train_data = np.load(self.dataset_train)
    #     X_train, y_train = train_data['x'], train_data['y']

    #     # Load test data
    #     test_data = np.load(self.dataset_test)
    #     X_test, y_test = test_data['x'], test_data['y']

    #     # Convert to PyTorch tensors
    #     X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    #     y_train = torch.tensor(y_train, dtype=torch.long)

    #     X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    #     y_test = torch.tensor(y_test, dtype=torch.long)

    #     return X_train, y_train, X_test, y_test

    # def train_model(self, num_epochs=2, batch_size=32, learning_rate=0.001):
    #     X_train, y_train, X_test, y_test = self.load_data()

    #     # Create DataLoader
    #     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    #     # Define loss function and optimizer
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    #     # Training loop
    #     for epoch in range(num_epochs):
    #         self.model.train()
    #         total_loss = 0

    #         for batch_X, batch_y in train_loader:
    #             optimizer.zero_grad()
    #             outputs = self.model(batch_X)
    #             loss = criterion(outputs, batch_y)
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()

    #         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    #     # Evaluate after training
    #     accuracy = self.evaluate_model(X_test, y_test, batch_size)
    #     self.accuracy_history_list.append(accuracy)

    # def evaluate_model(self, X_test, y_test, batch_size=32):
    #     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    #     self.model.eval()
    #     y_pred = []
    #     y_true = []

    #     with torch.no_grad():
    #         for batch_X, batch_y in test_loader:
    #             outputs = self.model(batch_X)
    #             predicted = torch.argmax(outputs, dim=1)
    #             y_pred.extend(predicted.cpu().numpy())
    #             y_true.extend(batch_y.cpu().numpy())

    #     accuracy = accuracy_score(y_true, y_pred)
    #     print(f"Model Accuracy: {accuracy:.4f}")
    #     return accuracy