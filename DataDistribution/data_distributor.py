import numpy as np
import tensorflow as tf
import os

def dirichlet_split(data, labels, alpha, num_splits):
    """
    Split the data and labels using Dirichlet distribution.
    """
    unique_classes = np.unique(labels)
    idx = {cls: np.where(labels == cls)[0] for cls in unique_classes}
    split_data = {i: [] for i in range(num_splits)}

    for cls in unique_classes:
        # Apply Dirichlet distribution to obtain proportions
        proportions = np.random.dirichlet([alpha] * num_splits)
        np.random.shuffle(idx[cls])
        
        # Split the indices based on proportions
        cls_splits = np.split(idx[cls], (np.cumsum(proportions[:-1]) * len(idx[cls])).astype(int))
        for i, split in enumerate(cls_splits):
            split_data[i].extend(split)
    
    # Separate data and labels for each split
    splits = [(data[indices], labels[indices]) for indices in split_data.values()]
    return splits

def main(client_id, alpha1, alpha2, save_dir):
    # Step 1: Load the full MNIST dataset
    mnist=tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_full = np.concatenate([x_train, x_test])
    y_full = np.concatenate([y_train, y_test])

    # Step 2: Create non-IIDness with Dirichlet (alpha1)
    splits = dirichlet_split(x_full, y_full, alpha=alpha1, num_splits=2)
    taken_part, _ = splits  # Use only the first part (50% of the dataset)

    # Step 3: Split taken_part further into training (30%) and testing (20%)
    train_part, test_part = dirichlet_split(taken_part[0], taken_part[1], alpha=alpha2, num_splits=2)

    # Save the datasets for this client
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, f"client_{client_id}_train.npz"),
                        x=train_part[0], y=train_part[1])
    np.savez_compressed(os.path.join(save_dir, f"client_{client_id}_test.npz"),
                        x=test_part[0], y=test_part[1])

    # Step 4: Print the digit distributions
    print(f"Client {client_id} - Training digit distribution: {np.bincount(train_part[1], minlength=10)}")
    print(f"Client {client_id} - Testing digit distribution: {np.bincount(test_part[1], minlength=10)}")

if __name__ == "__main__":
    alpha1 = 0.5  # Non-IIDness parameter for the first split
    alpha2 = 1  # Non-IIDness parameter for the second split
    save_dir = "./client_datasets"  # Directory to save client datasets

    # Repeat for 7 client IDs
    for client_id in range(1, 8):
        print(f"Processing data for Client {client_id}...")
        main(client_id, alpha1, alpha2, save_dir)
