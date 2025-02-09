import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes

    def forward(self, x, pen_flag=False):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 → 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 → 3x3
        x = self.pool(F.relu(self.conv4(x)))  # 3x3 → 1x1
        x = F.relu(self.conv5(x))  # Keep it 1x1

        x = x.view(-1, 256 * 2 * 2)  # Flatten
        x = F.relu(self.fc1(x))
        penultimate_output = F.relu(self.fc2(x))
        x = self.fc3(penultimate_output)

        if pen_flag:
            return x, penultimate_output
        return x


# class CNN5(nn.Module):
#     def __init__(self):
#         super(CNN5, self).__init__()
#         # Define layers
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
#         # Modify the fully connected layer input size
#         self.fc1 = nn.Linear(256 * 2 * 2, 128)  # Adjusted from 256 * 1 * 1
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)  # Assuming 10 classes for classification

#     def forward(self, x, pen_flag=False):
#         x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 14x14
#         x = self.pool(F.relu(self.conv2(x)))  # 14x14 → 7x7
#         x = self.pool(F.relu(self.conv3(x)))  # 7x7 → 3x3
#         x = self.pool(F.relu(self.conv4(x)))  # 3x3 → 1x1
#         x = F.relu(self.conv5(x))  # No pooling here, keeps size 1x1
        
#         x = x.view(-1, 256 * 2 * 2)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         penultimate_output = F.relu(self.fc2(x))
#         x = self.fc3(penultimate_output)
#         if pen_flag:
#             return x, penultimate_output
#         return x
