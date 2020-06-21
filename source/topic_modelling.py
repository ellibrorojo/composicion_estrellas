# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:05:08 2020

@author: javie
"""
import json
import pandas as pd
import gensim
import numpy as np
np.random.seed(0)
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim import models

def lemmatize_stemming(text):
    return PorterStemmer().stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
    
maxlineas = 10000

with open("../data/Electronics_5.json", "r") as f:
    counter = 0
    lines_reduced = []
 
    for line in f:
        line_dict = json.loads(line)
        if 'reviewText' in line_dict and 'overall' in line_dict and 'summary' in line_dict:
            lines_reduced.append({
                    'rating' : line_dict['overall'],
                    'summary': line_dict['summary'],
                    'review' : line_dict['reviewText']
                    })
            counter += 1
        if counter == maxlineas: break
    

df = pd.DataFrame(lines_reduced)


no_below = 2
no_above = 0.5
processed_docs = df['summary'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=100000)
   
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=5, chunksize=1000, passes=50)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))