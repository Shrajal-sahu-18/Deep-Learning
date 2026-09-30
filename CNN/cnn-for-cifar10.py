import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10

# Datasets and DataLoaders
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    c
])

trainset = CIFAR10(root = "./data",train = True, download = True, transform = transform)