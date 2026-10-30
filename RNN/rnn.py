import pandas as pd
import re

df = pd.read_csv("IMDB Dataset.csv")

df.isnull().sum()

df.drop_duplicates(inplace  = True) 

#Preprocessing
df["review"] = df["review"].str.lower()

# Remove Url's
def remove_urls(text):
    text = re.sub(r"http\S+" ,"",text) # (Pattern,replacement,string)
    return text
df["review"] = df["review"].apply(remove_urls)


# Remove Puncutations
def remove_punctuations(text):
    text = re.sub(r"[^A-Za-z0-9\s]","",text)# A-Z a-z 0-9 /s inko chhodke sabko remove karna hai.
    return text

df["review"] = df["review"].apply(remove_punctuations)