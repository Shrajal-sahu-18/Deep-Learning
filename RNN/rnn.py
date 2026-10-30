import pandas as pd
import re

df = pd.read_csv("IMDB Dataset.csv")

df.isnull().sum()

df.drop_duplicates(inplace  = True) 

#Preprocessing
df["review"] = df["review"].str.lower()

def remove_urls(text):
    text = re.sub(r"http\S+" ,"",text) # (Pattern,replacement,string)
    return text
df["review"] = df["review"].apply(remove_urls)