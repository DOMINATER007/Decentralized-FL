import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

class CNN3(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 28x28 MNIST -> 3x3 after pooling
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
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        penultimate_weights = x.clone()  # Clone penultimate layer output
        x = self.fc2(x)
        return x, penultimate_weights

    def get_penultimate_weights(self, x):
        with torch.no_grad():
            _, penultimate_weights = self.forward(x)
        return penultimate_weights

    # def train_model(self, data_path, epochs=1, batch_size=32, learning_rate=0.001):
    #     data = np.load(data_path)
    #     X_train, y_train = data['x'], data['y']
        
    #     X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dim
    #     print(f"\nInput shape before training: {X_train.shape}\n")  # Should print [batch_size, 1, 28, 28]

    #     y_train = torch.tensor(y_train, dtype=torch.long)

    #     train_dataset = TensorDataset(X_train, y_train)
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #     self.train()
    #     for epoch in range(epochs):
    #         total_loss = 0
    #         for inputs, targets in train_loader:
    #             #print(f"\nInputs.Shape --> {inputs.shape}\n")
    #             optimizer.zero_grad()
    #             outputs, _ = self.forward(inputs)
    #             loss = criterion(outputs, targets)
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
    #         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    #     # Calculate training accuracy
    #     y_pred, y_true = [], []
    #     self.eval()
    #     with torch.no_grad():
    #         for inputs, targets in train_loader:
    #             outputs, _ = self.forward(inputs)
    #             _, preds = torch.max(outputs, dim=1)
    #             y_pred.extend(preds.numpy())
    #             y_true.extend(targets.numpy())

    #     training_accuracy = accuracy_score(y_true, y_pred)
    #     return training_accuracy
    def train_model(self, data_path, epochs=1, batch_size=32, learning_rate=0.001):
        data = np.load(data_path)
        X_train, y_train = data['x'], data['y']
        
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dim
        print(f"\nInput shape before training: {X_train.shape}\n")  # Should print [batch_size, 1, 28, 28]

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
                outputs, penultimate = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # Calculate training accuracy and collect penultimate layer outputs
        y_pred, y_true = [], []
        penultimate_outputs = []
        self.eval()
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs, penultimate = self.forward(inputs)
                _, preds = torch.max(outputs, dim=1)
                y_pred.extend(preds.numpy())
                y_true.extend(targets.numpy())
                penultimate_outputs.extend(penultimate.numpy())  # Store penultimate layer outputs

        training_accuracy = accuracy_score(y_true, y_pred)
        print(np.array(penultimate_outputs).shape)
        md=self.get_model_details()
        print(f"\nModel Details : {md}\n")
        return {
            "training_accuracy": training_accuracy,
            "penultimate_outputs": np.array(penultimate_outputs),# Convert to NumPy for easy use
            "model_info":md
        }


    def test_model(self, data_path):
        data = np.load(data_path)
        X_test, y_test = data['x'], data['y']
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dim
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

        hidden_layers = []
        for name, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d)):
                layer_info = {
                    "name": name,
                    "type": layer.__class__.__name__,
                    "output_shape": list(layer.weight.shape) if hasattr(layer, "weight") else "N/A"
                }
                hidden_layers.append(layer_info)

        details = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "hidden_layers": len(hidden_layers)
        }
        return details


# Example usage:
# model = CNN3(input_channels=1, num_classes=10)
# training_accuracy = model.train_model("train_data.npz")
# print(f"Training Accuracy: {training_accuracy:.2f}")
# test_accuracy = model.test_model("test_data.npz")
# print(f"Test Accuracy: {test_accuracy:.2f}")
# print(model.get_model_details())
