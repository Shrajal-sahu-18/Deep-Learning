import pandas as pd
import numpy as np

df = pd.read_csv("powerplant_data.csv")

df.head(10)

df.isnull().sum()

X = df.drop(["PE"],axis = 1)
y = df["PE"]

# Split the data
from sklearn.model_selection import train_test_split