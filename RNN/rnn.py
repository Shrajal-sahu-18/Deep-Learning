import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

# Remove Html Tag
def removing_html(text):
    text = re.sub(r"<.*?>","",text)
    return text

df["review"] = df["review"].apply(removing_html)

#Remove Stopwords
def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_word = stopwords.words("english")

    for word in tokens:
        if word in stop_word:
            text = text.replace(word ,"")
    return text

df["review"] = df["review"].apply(remove_stopwords)