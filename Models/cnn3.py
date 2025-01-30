import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Assuming input images are 28x28 (e.g., MNIST)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Assuming 10 classes for classification

    def forward(self, x,pen_flag=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 3 * 3)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        penultimate_output = F.relu(self.fc2(x))
        x = self.fc3(penultimate_output)
        if pen_flag:
            return x,penultimate_output
        return x