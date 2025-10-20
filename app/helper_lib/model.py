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

class Critic(nn.Module):
        def __init__(self):        
            super(Critic, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
            self.batchnorm = nn.BatchNorm2d(128)
            self.act2 = nn.LeakyReLU(0.2, inplace=True)
            #self.drop2 = nn.Dropout(0.3)

            self.flatten = nn.Flatten()
            self.fc = nn.Linear(7 * 7 * 128, 1)


        def forward(self, x):
            x = self.conv1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = self.batchnorm(x)
            x = self.act2(x)
            #x = self.drop2(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

class Generator(nn.Module):
        def __init__(self, z_dim):
            super(Generator, self).__init__()
            self.z_dim = z_dim

            self.fc = nn.Linear(z_dim, 7 * 7 * 128)
            self.reshape = lambda x: x.view(x.size(0), 128, 7, 7)

            self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
            self.act1 = nn.ReLU(True) # nn.LeakyReLU(0.2, inplace=True)

            self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
            self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.fc(x)
            x = self.reshape(x)

            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.act1(x)

            x = self.deconv2(x)
            x = self.tanh(x)
            return x


def get_model(model_name):
    if model_name == "Homework2":
        return SimpleCNN()
    
    if model_name == "GAN":
         z_dim = 100
         gen = Generator(z_dim)
         critic = Critic()
         return gen,critic


