import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def perform_knowledge_distillation(leader, followers, epochs=5, lr=0.001, batch_size=32):
    """
    Perform Knowledge Distillation where followers learn from the leader's penultimate layer outputs.
    
    Args:
        leader: The leader client whose model acts as the teacher.
        followers: List of follower clients whose models act as students.
        epochs: Number of training epochs for KD.
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training.
    """
    # Define a loss function (e.g., Mean Squared Error for regression-like KD)
    criterion = nn.MSELoss()

    # Train each follower using the leader's penultimate layer outputs
    for follower in followers:
        follower_model = follower.model
        optimizer = optim.Adam(follower_model.parameters(), lr=lr)

        # Load dataset from .npz file
        dataset_path = follower.dataset_train
        data = np.load(dataset_path)
        train_data = data['data']  # Assuming 'data' contains the input features
        train_labels = data['labels']  # Assuming 'labels' contains the target labels

        # Convert to PyTorch tensors
        train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

        # Create a DataLoader for batching
        dataset = TensorDataset(train_data_tensor, train_labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"Training follower {follower.client_id} with leader {leader.client_id}...")

        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_data, _ in dataloader:  # Ignore labels since we're doing KD on penultimate outputs
                # Get penultimate layer outputs from the leader's model
                with torch.no_grad():
                    leader_penultimate_outputs = leader.get_penultimate_layer_outputs(batch_data)

                # Forward pass through the follower's model
                _, follower_penultimate_outputs = follower_model(batch_data)

                # Compute the KD loss
                loss = criterion(follower_penultimate_outputs, leader_penultimate_outputs)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        print(f"Follower {follower.client_id} trained successfully.")