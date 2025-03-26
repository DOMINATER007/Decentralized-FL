import numpy as np
import torch
import os
from torchvision import datasets, transforms

def dirichlet_split(data, labels, alpha, num_clients):
    """
    Split the dataset among `num_clients` using Dirichlet distribution to ensure class imbalance.
    """
    unique_classes = np.unique(labels)
    idx = {cls: np.where(labels == cls)[0] for cls in unique_classes}
    split_data = {i: [] for i in range(num_clients)}
    
    for cls in unique_classes:
        proportions = np.random.dirichlet([alpha] * num_clients)
        np.random.shuffle(idx[cls])
        
        # Compute split indices
        cls_splits = np.split(idx[cls], (np.cumsum(proportions[:-1]) * len(idx[cls])).astype(int))
        for i, split in enumerate(cls_splits):
            split_data[i].extend(split)
    
    # Extract data and labels for each client
    client_splits = {i: (data[indices], labels[indices]) for i, indices in split_data.items()}
    return client_splits

def data_prep(num_clients, alpha, save_dir):
    train_perc = 0.8   # Percentage of data allocated to
    # Load MNIST dataset using PyTorch
    transform = transforms.Compose([transforms.ToTensor()])
    full_mnist = datasets.MNIST(root="E:\MAJORPROJECT\Decentralized-FL\DataDistribution\data", train=True, download=True, transform=transform)
    
    x_full = full_mnist.data.numpy()  # Convert PyTorch tensors to NumPy arrays
    y_full = full_mnist.targets.numpy()
    
    # Apply Dirichlet split to allocate samples among clients
    client_splits = dirichlet_split(x_full, y_full, alpha=alpha, num_clients=num_clients)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for client_id, (x_client, y_client) in client_splits.items():
        print(f"\n*****{client_id+1}****\n")
        # Further split into train (80%) and test (20%)
        num_train = int(train_perc * len(x_client))
        indices = np.random.permutation(len(x_client))
        
        train_indices, test_indices = indices[:num_train], indices[num_train:]
        train_part = (x_client[train_indices], y_client[train_indices])
        test_part = (x_client[test_indices], y_client[test_indices])
        
        # Save dataset for this client
        np.savez_compressed(os.path.join(save_dir, f"client_{client_id+1}_train.npz"),
                            x=train_part[0], y=train_part[1])
        np.savez_compressed(os.path.join(save_dir, f"client_{client_id+1}_test.npz"),
                            x=test_part[0], y=test_part[1])

        print(f"Client {client_id} - Total samples: {len(x_client)}")
        print(f"Client {client_id} - Training samples: {len(train_part[0])}")
        print(f"Client {client_id} - Testing samples: {len(test_part[0])}")
        print(f"Client {client_id} - Training digit distribution: {np.bincount(train_part[1], minlength=10)}")
        print(f"Client {client_id} - Testing digit distribution: {np.bincount(test_part[1], minlength=10)}\n\n")

if __name__ == "__main__":
    num_clients = 7  # Number of clients
    alpha = 0.7  # Controls the class imbalance among clients
    save_dir = "./client_datasets"
    
    print("Processing data for all clients...")
    data_prep(num_clients, alpha, save_dir)
