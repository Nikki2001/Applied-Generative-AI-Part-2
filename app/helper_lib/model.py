import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 64x64x3 -> 64x64x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                  # halves H,W
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 32x32x16 -> 32x32x32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # After two pools: 64->32->16, so 32*16*16=8192, then 100 units
        self.fc2 = nn.Linear(100, 10)            # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # -> [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))     # -> [B, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16)             # Flatten
        x = F.relu(self.fc1(x))                  # 100 units
        x = self.fc2(x)                          # 10 classes
        return x

def get_model(model_name):
    if model_name == "Homework2":
        return SimpleCNN()


