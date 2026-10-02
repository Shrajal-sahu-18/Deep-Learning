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
            nn.MaxPool2D(2,2), # kernel = 2 ,stride = 2

            # Second Convonutional layer
            nn.conv2D(32,64,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2D(2,2), # kernel = 2 ,stride = 2


                # Third Layer
            nn.conv2D(64,128,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2D(2,2), # kernel = 2 ,stride = 2
        )
            

        
        self.fc_layers = nn.Sequential(
            nn.Linear(4*4*128,256),
            nn.ReLU(),

            nn.linear(256,10)
        )
        
    def forward(self,x):  #upper constructor hai call niche say hoga
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1) # Flattening
        x = self.fc_layers(x)
        return x

model =CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Trainning Our CNN
epochs = 10
trainning_loss = []
validation_loss = []
best_val_loss = float("inf")
for epoch in range(epochs):
    epoch_trainning_loss = 0.0

    for images,labels in trainloader:
        optimizer.zero_grad() # purane gradients clear
        output = model.forward(images) 
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        epoch_trainning_loss += loss.item()
    epoc_loss = epoch_trainning_loss /len(trainloader)
    trainning_loss.append(epoc_loss)

    model.eval()
    epoch_validloss = 0.0
    with torch.no_grad():
        for images,labels in testloader:
            outputs = model(images)
            loss = criterion(outputs,labels)
            epoch_validloss += loss.item()
        valid_loss = epoch_validloss / len(testloader)
        validation_loss.append(valid_loss)

    if best_val_loss > valid_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(),"best_model.pth")

# Evaluate Our Cnn
correct_labels = 0
total_labels = 0
model.eval()
with torch.no_grad():