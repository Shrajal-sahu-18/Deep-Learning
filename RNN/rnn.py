import pandas as pd

df = pd.read_csv("IMDB Dataset.csv")

df.isnull().sum()