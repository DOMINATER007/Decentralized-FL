import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

class CNN7(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN7, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # After three pooling operations
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        penultimate_weights = x.clone()  # Clone penultimate layer output
        x = self.fc2(x)
        return x, penultimate_weights

    def get_penultimate_weights(self, x):
        with torch.no_grad():
            _, penultimate_weights = self.forward(x)
        return penultimate_weights

    def train_model(self, data_path, epochs=1, batch_size=32, learning_rate=0.001):
        data = np.load(data_path)
        X_train, y_train = data['x'], data['y']
        
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dim
        print(f"\nInput shape before training: {X_train.shape}\n")

        y_train = torch.tensor(y_train, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs, _ = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # Calculate training accuracy
        y_pred, y_true = [], []
        self.eval()
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs, _ = self.forward(inputs)
                _, preds = torch.max(outputs, dim=1)
                y_pred.extend(preds.numpy())
                y_true.extend(targets.numpy())

        training_accuracy = accuracy_score(y_true, y_pred)
        return training_accuracy

    def test_model(self, data_path):
        data = np.load(data_path)
        X_test, y_test = data['x'], data['y']
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.long)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        y_pred, y_true = [], []
        self.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs, _ = self.forward(inputs)
                _, preds = torch.max(outputs, dim=1)
                y_pred.extend(preds.numpy())
                y_true.extend(targets.numpy())

        test_accuracy = accuracy_score(y_true, y_pred)
        return test_accuracy

    def get_model_details(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        details = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_architecture": str(self)
        }
        return details

# Example usage:
# model = CNN7(input_channels=1, num_classes=10)
# training_accuracy = model.train_model("train_data.npz")
# print(f"Training Accuracy: {training_accuracy:.2f}")
# test_accuracy = model.test_model("test_data.npz")
# print(f"Test Accuracy: {test_accuracy:.2f}")
# print(model.get_model_details())
