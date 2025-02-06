import json
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
        print("bind completed")
        self.server_address=self.server_socket.getsockname()
        self.server_port = self.server_socket.getsockname()[1]
        self.server_socket.listen(10)
        #self.server_thread = threading.Thread(target=self.receive_data_from_other_clients, daemon=True)
        print(self.server_port)
        #self.server_thread.start()
        #self.server_thread.join()
        self.data_of_others=None

    def json_encode1(self):
        data=dict()
        data['location']=self.coordinates
        data['id']=self.client_id
        obj=json.dumps(data)
        return obj
    
    def json_encode2(self):
        data = {
                        "model_architecture": str(self.model.__class__.__name__),
                        "coordinates": self.coordinates,
                        "bandwidth": self.bandwidth,
                        "accuracy_history_list": self.accuracy_history_list,
                        "client_id": self.client_id,
                    }
        
        obj=json.dumps(data)
        return obj        
    def braoadCast(self,peers):
        for addr,port in peers:
            self.server_socket.connect((addr,port))
            self.server_socket.send(self.json_encode1())
    def recvHelper(self):
        while True:
            client_soc,client_addr=self.server_socket.accept()
            try:
                data=client_soc.recv(1024)
                print(f"Received Data {data} from {client_addr}")
                self.data_of_others[self.client_id]=data
            except Exception as e:
                print(f"Excepton Occured {e}")
            finally:
                client_soc.close()
    def receive_end(self):
        listening_thread=threading.Thread(target=self.recvHelper)
        listening_thread.start()
    
