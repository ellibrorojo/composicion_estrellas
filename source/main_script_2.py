# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:34:20 2020

@author: javie
"""
import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from string import Formatter
from datetime import datetime
from matplotlib import pyplot as plt
import nltk
import networkx as nx
import seaborn as sns
import random
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics    
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import svm
import math as math
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from itertools import chain
from wordcloud import WordCloud
from collections import Counter
#import utils
import utils
import topics_holder as th

data_raw_0 = utils.electronics_5_to_raw_data_0(10000)

utils.perform_eda(data_raw_0, 'rating_distribution')
utils.perform_eda(data_raw_0, 'rating_distribution_item')
utils.perform_eda(data_raw_0, 'opinions_per_year')
utils.perform_eda(data_raw_0, 'opinions_per_item')
utils.perform_eda(data_raw_0, 'opinions_per_user')
utils.perform_eda(data_raw_0, 'rating_sd_per_item')
utils.perform_eda(data_raw_0, 'summary_review_length_comparison_per_rating')
utils.perform_eda(data_raw_0, 'text_length_per_rating')


data_raw_1 = utils.remove_rating_bias_from_raw_data(data_raw_0)

utils.perform_eda(data_raw_1, 'rating_distribution')
utils.perform_eda(data_raw_1, 'rating_distribution_item')
utils.perform_eda(data_raw_1, 'opinions_per_year')
utils.perform_eda(data_raw_1, 'opinions_per_item')
utils.perform_eda(data_raw_1, 'opinions_per_user')
utils.perform_eda(data_raw_1, 'rating_sd_per_item')
utils.perform_eda(data_raw_1, 'summary_review_length_comparison_per_rating')
utils.perform_eda(data_raw_1, 'text_length_per_rating')


odf = utils.execute_preprocessing_pipeline(data_raw_1)
odf = utils.execute_preprocessing_pipeline(data_raw_1, False)
odf = utils.load_latest_odf(nrows=214475, is_false=True)
odf = utils.load_latest_odf(nrows=17525, is_false=True)
odf = utils.load_latest_odf(nrows=1532805, is_false=True)


#tokens = utils.generate_bow(odf, False, show=False)
#bigrams = utils.extract_bigrams_from_bow(tokens_counter)

####################################################
####################################################

# Construimos un contador para almacenar en cuántas ocasiones aparece cada token
token_counter = Counter()
for index, row in odf.iterrows():
    token_counter.update(row['text'].split(', '))


print("Cantidad de tokens (distintos) antes de eliminar los poco frecuentes:", len(token_counter.keys()))
for word in list(token_counter):
    if token_counter[word] < 2:
        del token_counter[word]
print("Cantidad de tokens (distintos) tras eliminar los poco frecuentes:    ", len(token_counter.keys()))


#Construimos el diccionario
tokens_coded = {"":0, "UNK":1}
words = ["", "UNK"]
for word in token_counter:
    tokens_coded[word] = len(words)
    words.append(word) 





####################################################
####################################################



topics = th.generate_topics(token_counter)

topics_names = utils.get_all_topic_names(topics)

lista_resultados_analize = []
for topic_name in topics_names:
    res_fil, res_agg = utils.analize_wordset(odf, th.get_topic_by_name(topics, topic_name), False)
    lista_resultados_analize.append({'name':topic_name, 'resultados': res_fil, 'agregados': res_agg})


mat_doc_ws, mat_doc_ws_agg = utils.analize_wordset_occurrences(odf, lista_resultados_analize)

utils.visualize_wordsets_network_6(mat_doc_ws, group_size=5, k=0.3, ratings='F')

df_sharing = utils.get_sharing_matrix_2(mat_doc_ws)

utils.print_topic_heatmaps(mat_doc_ws)
utils.print_rating_heatmaps(mat_doc_ws)

popular_topic_combinations = utils.get_popular_topic_combinations(mat_doc_ws_agg, topics_names, 20)
utils.print_statistics_by_topic_heatmap_rating_value(mat_doc_ws)
utils.print_statistics_by_topic_heatmap_rating_sd(mat_doc_ws)

for item in lista_resultados_analize:
    utils.print_simple_histogram(item['agregados']['doc_number'], title=item['name'])

utils.get_respuesta_comprador(lista_resultados_analize, mat_doc_ws)




# PREDICCIÓN SOBRE TEXTO USANDO TREE/FOREST CLASSIFIER Y EVALUANDO CON RMSE
#clf_model_texto = DecisionTreeClassifier()
clf_model_texto = RandomForestClassifier()
cv = CountVectorizer()
#cv = TfidfVectorizer(ngram_range=(1, 2),stop_words = 'english')

X_texto = cv.fit_transform(odf['text'])
y_texto = odf['rating_0']

X_train_texto, X_test_texto, y_train_texto, y_test_texto = train_test_split(X_texto, y_texto, random_state=2020)
clf_model_texto_trained = clf_model_texto.fit(X_train_texto, y_train_texto)

utils.evaluar_modelo(clf_model_texto_trained, X_test_texto, y_test_texto)

utils.predict_from_text(clf_model_texto_trained, cv, 'This product is really bad')


# PREDICCIÓN SOBRE TEMAS USANDO TREE/FOREST CLASSIFIER Y EVALUANDO CON RMSE
#clf_model_temas = DecisionTreeClassifier()
clf_model_temas = RandomForestClassifier()

mat_doc_ws_2 = mat_doc_ws[mat_doc_ws['total_wordsets'] > 0]
docs_con_tema = list(mat_doc_ws_2.index.values)

X_temas = mat_doc_ws_2[topics_names]
y_temas = utils.map_rating(mat_doc_ws_2['rating'])

X_train_temas, X_test_temas, y_train_temas, y_test_temas = train_test_split(X_temas, y_temas, random_state=2020)
clf_model_temas_trained = clf_model_temas.fit(X_train_temas, y_train_temas)

utils.evaluar_modelo_reg(clf_model_temas_trained, X_test_temas, y_test_temas)




















# TRAS INTENTAR PROBAR FASTTEXT Y VER QUE SOLO VALE PARA MAC Y LINUX, BUSCAMOS ALTERNATIVAS:
#https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df
























########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################





    
tokens = utils.generate_bow(odf)

utils.busca_tokens(token_counter, ['work'])




