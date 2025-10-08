import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

torch.manual_seed(42) 
np.random.seed(42)

# Check which GPU is available
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Using device: {device}')

BATCH_SIZE = 64

# Prepare the Data & resize to 64x64
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

#torchvision.datasets.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

CLASSES = np.array(
    [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Parameters
NUM_CLASSES = 10
EPOCHS = 10 

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

model = SimpleCNN().to(device)
print(model)

from tqdm import tqdm

datalogs = []

#Save & Load Checkpoint
def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoints_cnn'):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(), #Contains & saves all the weights
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'cnn_epoch_{epoch:03d}.pth')
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, device):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']

        print(f"Loaded checkpoint from epoch {epoch}")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

        return epoch, loss, accuracy

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
best_accuracy = 0.0

for epoch in range(EPOCHS):
    running_loss = 0.0
    running_correct, running_total = 0, 0

    model.train()
    train_loader_with_progress = tqdm(iterable=train_loader, ncols=120, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
        inputs = inputs.to(device)
        labels = labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # predicted = torch.argmax(outputs.data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log data for tracking
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        running_loss += loss.item()  

        if (batch_number % 100 == 99):
            train_loader_with_progress.set_postfix({'avg accuracy': f'{running_correct/running_total:.3f}', 'avg loss': f'{running_loss/(batch_number+1):.4f}'})

            datalogs.append({
                "epoch": epoch + batch_number / len(train_loader), 
                "train_loss": running_loss / (batch_number + 1),
                "train_accuracy": running_correct/running_total,
            })

    datalogs.append({
        "epoch": epoch + 1, 
        "train_loss": running_loss / len(train_loader),
        "train_accuracy": running_correct/running_total,
    })

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = running_correct/running_total

    # Save checkpoint every epoch
    checkpoint_path = save_checkpoint(
        model, optimizer, epoch + 1, epoch_loss, epoch_accuracy, "C:/Users/nikki/sps_genai/app/helper_lib"
    )

    # Save best model
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_path = save_checkpoint(
            model, optimizer, epoch + 1, epoch_loss, epoch_accuracy, 
            checkpoint_dir='checkpoints_cnn/best'
        )
        print(f"New best model saved! Accuracy: {epoch_accuracy:.2f}%")
print("Finished Training")