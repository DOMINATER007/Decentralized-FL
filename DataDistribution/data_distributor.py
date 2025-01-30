import numpy as np
import torch
import os
from torchvision import datasets, transforms

def dirichlet_split(data, labels, alpha, num_splits):
    """
    Split the dataset into `num_splits` parts using Dirichlet distribution.
    """
    unique_classes = np.unique(labels)
    idx = {cls: np.where(labels == cls)[0] for cls in unique_classes}
    split_data = {i: [] for i in range(num_splits)}

    for cls in unique_classes:
        proportions = np.random.dirichlet([alpha] * num_splits)
        np.random.shuffle(idx[cls])

        # Compute split indices
        cls_splits = np.split(idx[cls], (np.cumsum(proportions[:-1]) * len(idx[cls])).astype(int))
        for i, split in enumerate(cls_splits):
            split_data[i].extend(split)

    # Extract data and labels for each split
    splits = [(data[indices], labels[indices]) for indices in split_data.values()]
    return splits

def main(client_id, alpha1, alpha2, save_dir):
    # Load MNIST dataset using PyTorch
    transform = transforms.Compose([transforms.ToTensor()])
    full_mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    x_full = full_mnist.data.numpy()  # Convert PyTorch tensors to NumPy arrays
    y_full = full_mnist.targets.numpy()

    # Step 1: Apply first Dirichlet split (alpha1)
    splits = dirichlet_split(x_full, y_full, alpha=alpha1, num_splits=2)
    taken_part, _ = splits  # Use only the first partition

    # Step 2: Further split into train (30%) and test (20%) using alpha2
    train_part, test_part = dirichlet_split(taken_part[0], taken_part[1], alpha=alpha2, num_splits=2)

    # Step 3: Save dataset for this client
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, f"client_{client_id}_train.npz"),
                        x=train_part[0], y=train_part[1])
    np.savez_compressed(os.path.join(save_dir, f"client_{client_id}_test.npz"),
                        x=test_part[0], y=test_part[1])

    # Print the digit distribution per client
    print(f"Client {client_id} - Training digit distribution: {np.bincount(train_part[1], minlength=10)}")
    print(f"Client {client_id} - Testing digit distribution: {np.bincount(test_part[1], minlength=10)}")

if __name__ == "__main__":
    alpha1 = 0.5  # Controls how non-IID the first split is
    alpha2 = 1.0  # Controls train-test split randomness
    save_dir = "./client_datasets"

    for client_id in range(1, 8):
        print(f"Processing data for Client {client_id}...")
        main(client_id, alpha1, alpha2, save_dir)
