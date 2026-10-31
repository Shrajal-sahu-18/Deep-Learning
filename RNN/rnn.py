import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset , DataLoader

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

# Steeming
def stemming(text):
    ps = PorterStemmer()
    stemmed_words = []

    tokens = word_tokenize(text)
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)

    return " ".join(stemmed_words)

df["review"] = df["review"].apply(stemming)

# Encoding
le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])

y = df["sentiment"]

# Vecotrization
tf = TfidfVectorizer(max_features = 5000)  # mtlb top ke 5000 words chahiye kyuki agar max_feasture nhi karenge to jyada word ke karan hamara system isko handle nhi kar payega
X = tf.fit_transform(df["review"])

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.2,random_state = 42)

X_train = X_train.toarray()
X_test = X_test.toarray()

trainset = TensorDataset(
    torch.from_numpy(X_train).float(), # Tensor may Convert
    torch.from_numpy(y_train.values).float()
)

testset = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test.values).float()
)

train_loader = DataLoader(trainset , shuffle = True , batch_size = 64)
test_loader = DataLoader(testset , shuffle = True , batch_size = 64)