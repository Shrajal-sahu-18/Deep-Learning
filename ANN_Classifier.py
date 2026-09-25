import pandas as pd
import numpy as np

df = pd.read_csv("DateFruit_Dataset.csv")

X = df.drop("Class",axis = 1)
y = df["Class"]