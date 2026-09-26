import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("DateFruit_Dataset.csv")

X = df.drop("Class",axis = 1)
y = df["Class"]

y.unique()

from sklearn.preprocessing import LabelEncoder,StandardScaler 

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.2,random_state = 42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled,dtype = torch.float32)
X_test_tensor = torch.tensor(X_test_scaled,dtype = torch.float32)

y_train_tensor = torch.tensor(y_train,dtype = torch.long)
y_test_tensor = torch.tensor(y_test,dtype = torch.long)

train_dataset = TensorDataset(X_train_tensor , y_train_tensor)
test_dataset = TensorDataset(X_test_tensor,y_test_tensor)