import pandas as pd

df = pd.read_csv("IMDB Dataset.csv")

df.isnull().sum()

df.drop_duplicates(inplace  = True) 