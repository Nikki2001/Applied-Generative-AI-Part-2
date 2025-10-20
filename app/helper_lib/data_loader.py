import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label (not used in GANs)

def get_data_loader(data_dir, batch_size=32, train=True, download = True, transform=None, dataset=None):
    # TODO: create the data loader
    if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

    if dataset == "CIFAR10":
         dataset = datasets.CIFAR10(
              root = data_dir, train=train, download = download, transform=transform
         )
         loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
         return loader

    if dataset == "MNIST":
         dataset = datasets.MNIST(
              root = data_dir, train=train, download = download, transform=transform
         )
         loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
         print("LOADER")
         return loader

    dataset = CustomImageDataset(root_dir=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return loader
