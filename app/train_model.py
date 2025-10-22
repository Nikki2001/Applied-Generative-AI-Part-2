
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.data_loader import get_data_loader


device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


dataloader = get_data_loader('./data', batch_size=32, train=True, download= True, transform=transform, dataset="MNIST")
generator,critic = get_model('GAN')
train_gan(generator, critic, dataloader, device, epochs=10) 