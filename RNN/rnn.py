import pandas as pd
import re

df = pd.read_csv("IMDB Dataset.csv")

df.isnull().sum()

df.drop_duplicates(inplace  = True) 

#Preprocessing
df["review"] = df["review"].str.lower()