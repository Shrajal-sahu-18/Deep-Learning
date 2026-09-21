import pandas as pd
import numpy as np

df = pd.read_csv("powerplant_data.csv")

df.head(10)

df.isnull().sum()

X = df.drop(["PE"],axis = 1)
y = df["PE"]

# Split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.2, random_state = 42
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import torch
import torch.nn as nn

X_train_tensor = torch.tensor(X_train_scaled , dtype = torch.float32)
X_test_tensor = torch.tensor(X_test_scaled , dtype = torch.float32)
y_train_tensor = torch.tensor(y_train.values,dtype = torch.float32).view(-1,1)
y_test_tensor = torch.tensor(y_test.values ,dtype = torch.float32).view(-1,1)

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

train_loader = DataLoader(train_dataset , batch_size = 32 , shuffle = True)
test_loader = DataLoader(test_dataset , batch_size = 32)

# Define our ANN Model
class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()

        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(X_train.shape[1],6),
            nn.ReLU(),

            # 2nd Layer
            nn.Linear(6,6),
            nn.ReLU(),

            # OutputLayer

            nn.Linear(6,1),
        )
    def forward(self,X):
        return self.model(X)

import torch.optim as optim

model = ANN()

#loss ,optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

#Train the ANN
training_losses = []
vaild_loss = []
epochs = 100

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for xb,yb in train_loader:
        # Xb  = feature of one batch 
        # yb  = labels of one batch (target column)
        optimizer.zero_grad() # optimizer gradient ko y accumalate kar leta isliye har step may new gradient calculate hae hamare new batch ke liye new gradient calculate hae isliye gradient ko zero kar rahe hai 
        outputs = model(xb) #Forward propgation step
        loss = criterion(outputs,yb) # compute loss loss for one batch then second batch
        loss.backward() # back prop compute gradients
        optimizer.step() # params update
        running_loss += loss.item()  # ye loss pytorch ne calculate kiya hai isliye ye ek tensor value hai hame isko python float value may convert karna padega
    epoch_train_loss = running_loss / len(train_loader)
    training_losses.append(epoch_train_loss)



        # vaildation
    
    running_val_loss = 0.0
    with torch.no_grad(): # pytorch by default gardient calculate karta hai yaha hmm bata rahe hai ki koi grad calculate nhi karna hai kyuki hamaramodel already train ho chuka hai 
        for xb ,yb in test_loader:
            outputs = model(xb)
            loss = criterion(outputs,yb)
            running_val_loss += loss# auto grad automatic gradient calculate
    epoch_val_loss = running_val_loss / len(test_loader)
    vaild_loss.append(epoch_val_loss)

    print(f"epoch ${epoch + 1} / {epochs} ==> train loss = $ {epoch_train_loss} & vaild loss = $ {epoch_val_loss}")