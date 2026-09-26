import pandas as pd
import numpy as np

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