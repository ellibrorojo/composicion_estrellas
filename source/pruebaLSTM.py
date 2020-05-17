# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:35:41 2020

@author: Javier
"""

#library imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
#import jovian
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import utils

#input
x = torch.tensor([[1,2, 12,34, 56,78, 90,80],
                 [12,45, 99,67, 6,23, 77,82],
                 [3,24, 6,99, 12,56, 21,22]])


model1 = nn.Embedding(100, 7, padding_idx=0)
model2 = nn.LSTM(input_size=7, hidden_size=3, num_layers=1, batch_first=True)



out1 = model1(x)
out2 = model2(out1)

print(out1.shape)
print(out1)


out, (ht, ct) = model2(out1)
print(ht)



model3 = nn.Sequential(nn.Embedding(100, 7, padding_idx=0),
                        nn.LSTM(input_size=7, hidden_size=3, num_layers=1, batch_first=True))



out, (ht, ct) = model3(x)
print(out)









#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################


def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]
    

def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

def train_model(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


#LSTM with fixed length input

class LSTM_fixed_len(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

def predict_from_text(text):
    prueba_ds = ReviewsDataset([np.array(encode_sentence('Absolutely wonderful - silky and sexy and comfortable', vocab2index))], [3])
    
    for x,y,l in DataLoader(prueba_ds, batch_size=batch_size, shuffle=True):
        print(torch.max(model_fixed(x.long()), 1)[1])
        
#loading the data
reviews = pd.read_csv("reviews.csv")
#print(reviews.shape)
#reviews.head()


reviews['Title'] = reviews['Title'].fillna('')
reviews['Review Text'] = reviews['Review Text'].fillna('')
reviews['review'] = reviews['Title'] + ' ' + reviews['Review Text']


#keeping only relevant columns and calculating sentence lengths
reviews = reviews[['review', 'Rating']]
reviews.columns = ['review', 'rating']
reviews['review_length'] = reviews['review'].apply(lambda x: len(x.split()))
#reviews.head()

#changing ratings to 0-numbering
zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])

#mean sentence length
#np.mean(reviews['review_length'])


#tokenization
tok = spacy.load('en')

#count number of occurences of each word
counts = Counter()
for index, row in reviews.iterrows():
    counts.update(tokenize(row['review']))
    
    
    
#deleting infrequent words
print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:",len(counts.keys()))
    
    
#creating vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)    
    
    
reviews['encoded'] = reviews['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index)))
#reviews.head()

#check how balanced the dataset is
Counter(reviews['rating'])

X = list(reviews['encoded'])
y = list(reviews['rating'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_valid, y_valid)


batch_size = 5000
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)



model_fixed =  LSTM_fixed_len(vocab_size=vocab_size, embedding_dim=50, hidden_dim=50)
train_model(model_fixed, epochs=30, lr=0.01)



prueba_ds = ReviewsDataset([np.array(encode_sentence('Absolutely wonderful - silky and sexy and comfortable', vocab2index))], [3])

for x,y,l in DataLoader(prueba_ds, batch_size=batch_size, shuffle=True):
    print(torch.max(model_fixed(x.long()), 1)[1])



###################################################################################################################
    
    
    

reviews = odf.copy()





#keeping only relevant columns and calculating sentence lengths
reviews = reviews[['text', 'rating']]
reviews['review_length'] = reviews['text'].apply(lambda x: len(x.split()))

#changing ratings to 0-numbering
zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])

tok = spacy.load('en')

#count number of occurences of each word
counts = Counter()
for index, row in reviews.iterrows():
    counts.update(tokenize(row['text']))
    
    
    
#deleting infrequent words
print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:",len(counts.keys()))
    
    
#creating vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)    
    
    
reviews['encoded'] = reviews['text'].apply(lambda x: np.array(encode_sentence(x, vocab2index)))

Counter(reviews['rating']) #check how balanced the dataset is

X = list(reviews['encoded'])
y = list(reviews['rating'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_valid, y_valid)


batch_size = 5000
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)



model_fixed =  LSTM_fixed_len(vocab_size=vocab_size, embedding_dim=50, hidden_dim=50)
train_model(model_fixed, epochs=5, lr=0.01)

utils.evaluar_modelo_lstm(model_fixed, val_dl)








