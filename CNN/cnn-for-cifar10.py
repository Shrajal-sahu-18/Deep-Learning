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
testset = CIFAR10(root = "./data",train = False , download = True , transform = transform)

trainloader = DataLoader(trainset,batch_size = 64,shuffle = True)
testloader = DataLoader(testset,batch_size = 64)

## Build The Cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv_layers = nn.Sequential(
            # First Convonutional Layer
            nn.conv2D(3,32,kernel_size =3,padding = 1),
            nn.Relu(),
            nn.MaxPool2D(2,2) # kernel = 2 ,stride = 2

            # Second Convonutional layer
            nn.conv2D(32,64,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2D(2,2) # kernel = 2 ,stride = 2


              # Third Layer
            nn.conv2D(64,128,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2D(2,2) # kernel = 2 ,stride = 2