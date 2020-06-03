# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:34:20 2020

@author: javie
"""
import json
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
#import math as math
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from string import punctuation
from sklearn import svm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import ngrams
from itertools import chain
from wordcloud import WordCloud
from sklearn import metrics    
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import re
import spacy
#import jovian
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error


'''
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.help.upenn_tagset('WP$')
'''
########################################################################################################################
def parse(path):
  g = open(path, "r")
  for l in g:
    yield json.loads(l)
########################################################################################################################
def getDF(path, nrows):
    #timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN GETDF', timestamps=[])
    if nrows == 9999999999:
        nrows = 6737497
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        #if i % 500000 == 0:
            #timestamps = calcula_y_muestra_tiempos('BUCLE RAW DATA: i='+str(i)+' de '+str(nrows), timestamps)
        if i == nrows:
            break
    #timestamps = calcula_y_muestra_tiempos('FIN FUNCIÓN GETDF', timestamps)
    return pd.DataFrame.from_dict(df, orient='index')
########################################################################################################################
def electronics_5_to_raw_data_0(nrows=9999999999):
    timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN ELECTRONICS_5_TO_RAW_DATA_0', timestamps=[])
    df = getDF(r'..\data\Electronics_5.json', nrows)
    timestamps = calcula_y_muestra_tiempos('SE PROCEDE A CREAR EL CAMPO OPINION_YEAR', timestamps)
    df['reviewTime'] = df.apply(lambda row: datetime.utcfromtimestamp(int(row['unixReviewTime'])).strftime('%Y%m%d'), axis=1)
    df['opinion_year'] = df.apply(lambda row: row['reviewTime'][:-4], axis=1)
    timestamps = calcula_y_muestra_tiempos('SE PROCEDE A ELIMINAR COLUMNAS QUE NO USAREMOS', timestamps)
    df = df.drop(columns=['vote', 'verified', 'style', 'reviewerName', 'unixReviewTime', 'image', 'reviewTime'])
    timestamps = calcula_y_muestra_tiempos('SE PROCEDE A RENOMBRAR ALGUNAS COLUMNAS', timestamps)
    df = df.rename(columns={'overall':'rating', 'reviewerID':'user_id', 'asin':'item_id', 'reviewText':'review'})
    timestamps = calcula_y_muestra_tiempos('SE PROCEDE A SUSTITUIR LOS NANS POR CADENAS VACÍAS', timestamps)
    df = df.fillna('')
    calcula_y_muestra_tiempos('FIN FUNCIÓN ELECTRONICS_5_TO_RAW_DATA_0', timestamps)
    
    return df
########################################################################################################################
def print_simple_histogram(data_agg, labels=None, title='Title'):    
    x_values = list(data_agg.index.values)
    x_values = list(map(int, x_values))
    fig, ax = plt.subplots()
    rotation = 0
    if len(x_values) > 5:
        rotation = 45
    if labels != None:
        ax.set_xticklabels(labels, rotation=rotation, ha='center')
    ax.set_xticks(x_values)
    ax.get_xticks()
    ax.bar(x_values, data_agg, color='tab:blue')
    plt.xticks(rotation=rotation, ha='center')
    plt.title(title)
    plt.show()
########################################################################################################################
def print_simple_boxplot(data, labels, yscale, grid, title='Title'):
    fig, ax = plt.subplots()
    ax.set_yscale(yscale)
    if grid != 'None':
        ax.yaxis.grid(True, which = grid)
    ax.set_xticklabels(labels)
    ax.boxplot(data)
    plt.title(title)
    plt.show()
########################################################################################################################
def strfdelta(tdelta, fmt='{H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
#def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):

    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)
########################################################################################################################
def calcula_y_muestra_tiempos(texto, timestamps):
    timestamps.append(datetime.now())
    tot = len(timestamps)
    
    is_inicio = ['INICIO'].count(texto.split(' ')[0]) > 0
    is_fin    = ['FIN'].count(texto.split(' ')[0]) > 0
    embellecedor = ' ____ ' if is_inicio or is_fin else ' :::: '

    if is_fin:
        print((strfdelta(timestamps[tot-1]-timestamps[0])+embellecedor+texto+' ').ljust(100, '_'))
    else:
        print((strfdelta(timestamps[tot-1]-timestamps[tot-2])+embellecedor+texto+' ').ljust(100, '_'))
        return timestamps
########################################################################################################################
def print_stacked_histogram(data_agg, bottom_colname, bottom_legend, top_colname, top_legend, title='Title'):
    x_values = list(data_agg.index.values)
    x_values = list(map(int, x_values))
    plt_bottom  = plt.bar(x_values, data_agg[bottom_colname], color='tab:red')
    plt_top     = plt.bar(x_values, data_agg[top_colname]-data_agg[bottom_colname], bottom=data_agg[bottom_colname], color='tab:blue')
    plt.legend((plt_bottom[0], plt_top[0]), (bottom_legend, top_legend))
    plt.title(title)
    plt.show()
########################################################################################################################
def remove_rating_bias_from_raw_data(raw_data, seed=2020):
    totals_by_rating = raw_data.groupby('rating').agg({'summary': np.size}).rename(columns={'summary':'total_docs_rating'})
    min_docs = min(totals_by_rating['total_docs_rating'])
    raw_data_1 = raw_data[raw_data['rating']==1].sample(n=min_docs, random_state=seed)
    raw_data_2 = raw_data[raw_data['rating']==2].sample(n=min_docs, random_state=seed)
    raw_data_3 = raw_data[raw_data['rating']==3].sample(n=min_docs, random_state=seed)
    raw_data_4 = raw_data[raw_data['rating']==4].sample(n=min_docs, random_state=seed)
    raw_data_5 = raw_data[raw_data['rating']==5].sample(n=min_docs, random_state=seed)
    raw_data = raw_data_1.append(raw_data_2).append(raw_data_3).append(raw_data_4).append(raw_data_5)
    raw_data = raw_data.sort_index()
    return raw_data
########################################################################################################################
def add_lenght_info(raw_data):
    raw_data['summary_length'] = raw_data.apply(lambda row: len(row['summary'].split(' ')), axis=1)
    raw_data['review_length'] = raw_data.apply(lambda row: len(row['review'].split(' ')), axis=1)
    raw_data['total_text'] = raw_data.apply(lambda row: row['summary_length']+row['review_length'], axis=1)
    return raw_data
########################################################################################################################
def perform_eda(raw_data, analysis):
    if analysis == 'opinions_per_year':
        print_simple_histogram(raw_data.groupby('opinion_year').count()['item_id'], title='Cantidad de opiniones publicadas por año')
    
    elif analysis == 'opinions_per_item':
        this_data = raw_data.groupby('item_id').agg({'rating':np.size})['rating']    
        print('Aparecen ' + str(len(this_data)) + ' artículos distintos.')
        plt.hist(this_data, bins=50, log=True, color='tab:blue')
        plt.title('Cantidad de artículos por número de opiniones')
        print_simple_boxplot(data=this_data, labels=[''], yscale='log', grid='minor', title='Artículos por número de opiniones')
    
    elif analysis == 'opinions_per_user':
        this_data = raw_data.groupby('user_id').agg({'rating':np.size})['rating']
        print('Aparecen ' + str(len(this_data)) + ' usuarios distintos.')
        plt.hist(this_data, bins=50, log=True, color='tab:blue')
        plt.title('Cantidad de usuarios por número de opiniones')
        print_simple_boxplot(data=this_data, labels=[''], yscale='log', grid='minor', title='Usuarios por número de opiniones')
            
    elif analysis == 'rating_sd_per_item':
        this_data = raw_data.groupby('item_id').agg({'rating':np.std})
        this_data = this_data.dropna()['rating'] #Esta línea sobra si se toman todas las opiniones, pues no habrá nans
        print_simple_boxplot(data=this_data, labels=[''], yscale='linear', grid='major', title='Dispersión de la puntuación')
        
    elif analysis == 'rating_distribution':
        this_data = raw_data.groupby('rating').agg({'summary':np.size}).rename(columns={'summary':'total_docs'})['total_docs']
        print_simple_histogram(data_agg=this_data, title='Distribución de opiniones por puntuación')
        
    elif analysis == 'rating_distribution_item':
        d = round(raw_data.groupby('item_id').agg({'rating':np.median}))
        d['item_id'] = d.index.values
        this_data = d.groupby('rating').count()['item_id']
        print_simple_histogram(data_agg=this_data, title='Distribución de opiniones por puntuación mediana de artículo')
        
    elif analysis == 'summary_review_length_comparison_per_rating':
        raw_data = add_lenght_info(raw_data)
        raw_data['ratio_text_summary'] = raw_data.apply(lambda row: row['total_text']/row['summary_length'], axis=1)
        resultados_agregados = raw_data.groupby('rating').agg({'summary_length':np.mean, 'review_length':np.mean, 'ratio_text_summary':np.mean, 'total_text':np.mean})
        print_stacked_histogram(resultados_agregados, 'summary_length', '# palabras en summary', 'review_length', '# palabras en review', title='Tamaños review y summary')
        #print_stacked_histogram(resultados_agregados, 'review_length', '# palabras en review', 'summary_length', '# palabras en summary')
        lista_de_ratios = []
        lista_de_ratios.append(raw_data[raw_data.rating==1]['ratio_text_summary'])
        lista_de_ratios.append(raw_data[raw_data.rating==2]['ratio_text_summary'])
        lista_de_ratios.append(raw_data[raw_data.rating==3]['ratio_text_summary'])
        lista_de_ratios.append(raw_data[raw_data.rating==4]['ratio_text_summary'])
        lista_de_ratios.append(raw_data[raw_data.rating==5]['ratio_text_summary'])
        print_simple_boxplot(data = lista_de_ratios, labels=[1, 2, 3, 4, 5], yscale='log', grid='minor', title='Proporción tamaño review/summary')
    
    elif analysis == 'text_length_per_rating':
        raw_data = add_lenght_info(raw_data)
        lista_de_longitudes = []
        lista_de_longitudes.append(raw_data[raw_data.rating==1]['total_text'])
        lista_de_longitudes.append(raw_data[raw_data.rating==2]['total_text'])
        lista_de_longitudes.append(raw_data[raw_data.rating==3]['total_text'])
        lista_de_longitudes.append(raw_data[raw_data.rating==4]['total_text'])
        lista_de_longitudes.append(raw_data[raw_data.rating==5]['total_text'])
        print_simple_boxplot(data = lista_de_longitudes, labels=[1, 2, 3, 4, 5], yscale='log', grid='minor', title='Longitud del texto completo')
########################################################################################################################
def build_odf(doc_numbers, ratings, summaries_bigrams, summaries_raw, reviews_bigrams, reviews_raw, corpus_aceptado=None):
    timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN BUILD_ODF', timestamps=[])
    text_list_of_strings = []
    text = []
    text2 = []
    for i in range(0, len(summaries_bigrams)):
        text.append(summaries_bigrams[i]+reviews_bigrams[i])
        if i%200000 == 0:
           timestamps = calcula_y_muestra_tiempos('BUCLE GENERACIÓN VARIABLE TEXT: '+str(i)+' DE '+str(len(summaries_bigrams)), timestamps)
    timestamps = calcula_y_muestra_tiempos('VARIABLE TEXT GENERADA', timestamps)
    
    if corpus_aceptado is not None and len(corpus_aceptado) > 0:
        timestamps = calcula_y_muestra_tiempos('SE HA PROPORCIONADO CORPUS_ACEPTADO', timestamps)
        timestamps = calcula_y_muestra_tiempos('SE PROCEDE A GENERERAR LA VARIABLE TEXT APLICANDDO CORPUS_ACEPTADO', timestamps)
        corpus_aceptado = list(corpus_aceptado['bigram'])
        i = 0
        for linea in text:
            linea2 = []
            for item in linea:
                if item in corpus_aceptado:
                    linea2.append(item)
            text2.append(linea2)
            i += 1
            if i%200000 == 0:
                timestamps = calcula_y_muestra_tiempos('BUCLE GENERACIÓN VARIABLE TEXT: '+str(i)+' DE '+str(len(text)), timestamps)
        timestamps = calcula_y_muestra_tiempos('VARIABLE TEXT GENERADA', timestamps)
        for linea2 in text2:
            text_list_of_strings.append(', '.join(map(str, linea2)))
        timestamps = calcula_y_muestra_tiempos('VARIABLE TEXT TRANSFORMADA', timestamps)
    else:
        for linea in text:
            text_list_of_strings.append(', '.join(map(str, linea)))
        timestamps = calcula_y_muestra_tiempos('VARIABLE TEXT TRANSFORMADA', timestamps)
    
    summary_tokens_length = [len(x) for x in summaries_bigrams]
    
    tmp = pd.DataFrame(zip(doc_numbers, ratings, text_list_of_strings, summaries_raw, reviews_raw, summary_tokens_length), columns=['doc_number', 'rating', 'text', 'summary_raw', 'review_raw', 'summary_tokens_length'])
    tmp.index = tmp['doc_number']
    
    retorno = tmp[['doc_number', 'rating', 'text', 'summary_raw', 'review_raw', 'summary_tokens_length']]
    retorno['rating_0'] = map_rating(retorno['rating'])
    calcula_y_muestra_tiempos('FIN FUNCIÓN BUILD_ODF', timestamps)
    
    return retorno
########################################################################################################################
def sent_to_words(sentences):
    retorno = []
    for sentence in sentences:
        retorno.append(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations     
    return retorno
########################################################################################################################
def remove_stopwords(texts):
    stop_words = get_stopwords('pre')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
########################################################################################################################
def make_bigrams(texts, data_words):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=3, threshold=3) # higher threshold fewer phrases.
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]
########################################################################################################################
def check_existencia (conjunto, tokens_a_buscar):
    for token in tokens_a_buscar:
        if token in conjunto:
            return True
########################################################################################################################
def analize_wordset(df, wordset_wrapper, show=False):
    
    timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN ANALIZE_WORDSET', timestamps=[])
    
    max_distance = 3
    wordset_ands = wordset_wrapper['wordset']['ands']
    wordset_ors = wordset_wrapper['wordset']['ors']
    wordset_name = wordset_wrapper['name']
    timestamps = calcula_y_muestra_tiempos('WORDSET A ANALIZAR: ' + wordset_name, timestamps)
    lista_or_words = get_ws_or_words(wordset_wrapper)
    rows = []
    
    elementos_or = get_wsw_structure(wordset_wrapper)[1]
    
    #timestamps = calcula_y_muestra_tiempos('ARRANCA EL BUCLE DE OPINIONES', timestamps)
    ##################################################_C_O_R_E_########################################
    i = 0
    j = 0
    for opinion in df.iterrows():
        ors_compliance = False
        and_compliance = False
        parte = 'summary'
        opinion_text_full = opinion[1]['text'].split(', ')
        opinion_text_set = set(opinion_text_full)
        
        if (len(wordset_ands) == 0 or check_existencia(opinion_text_set, wordset_ands)) and (elementos_or == 0 or check_existencia(opinion_text_set, lista_or_words)): # si no hay ninguna opcion de hacer hit no entramos en loop
            for pase in range(0,2):
                if parte == 'summary':
                    opinion_text = opinion_text_full[0:opinion[1]['summary_tokens_length']]
                    parte = 'review'
                elif not ors_compliance:
                    opinion_text = opinion_text_full[opinion[1]['summary_tokens_length']:]
        
                if not ors_compliance:
                    if not and_compliance:
                        ands_count = 0
                        for and_word in wordset_ands:
                            if opinion_text.count(and_word) > 0:
                                ands_count += 1
                                
                        and_compliance = ands_count == len(wordset_ands)
                    if and_compliance: # se cumplen los ands
                        for or_group in wordset_ors:
                            if not ors_compliance:
                                syn0_words = or_group['syn0']
                                syn0_indices = []
                                
                                for syn0_word in syn0_words:
                                    syn0_indices.extend([i for i, j in enumerate(opinion_text) if j == syn0_word])
                                    
                                syn1_words = or_group['syn1']
                                syn1_indices = []
                                for syn1_word in syn1_words:
                                    syn1_indices.extend([i for i, j in enumerate(opinion_text) if j == syn1_word])
                            
                                if len(syn0_indices) == 0 and len(syn1_indices) == 0: #ESTE OR_GROUP NO SATISFACE
                                    continue
                                elif len(syn0_indices) > 0 and len(syn1_indices) == 0: #LOS SYN0 PUEDEN SATISFACER, NO ASI LOS SYN1
                                    # CONSTRUCCION ARRAY DE SYN0
                                    min_posicion = max(min(syn0_indices)-max_distance, 0)
                                    max_posicion = min(max(syn0_indices)+max_distance, len(opinion_text))
                                    tokens_implicados = opinion_text[min_posicion:max_posicion+1]
                                    syn0_array = [0] * len(tokens_implicados)
                                    syn0_indices = [x-min_posicion for x in syn0_indices]
                                    for syn0_indice in syn0_indices:
                                        syn0_array[syn0_indice] = 1
                                    # CONSTRUCCION ARRAY DE NOTS
                                    nots_words = wordset_ors[0]['nots']
                                    nots_indices = []
                                    for nots_word in nots_words:
                                        nots_indices.extend([i for i, j in enumerate(tokens_implicados) if j == nots_word])
                                    if len(nots_indices) == 0: # NO HAY NOTS, POR LO QUE EL OR_GROUP SATISFACE Y LA OPINION SE ACEPTA
                                        ors_compliance = True
                                        break
                                    else:
                                        nots_array = [0] * len(tokens_implicados)
                                        for nots_indice in nots_indices:
                                            nots_array[nots_indice] = 1
                                        # CONSTRUCCION MATRIZ
                                        #matrix = pd.DataFrame(zip(syn0_array, nots_array)).transpose()
                                        #matrix.columns = tokens_implicados
                                        #matrix.index = ['syn0', 'nots']
                                        
                                        for syn0_indice in syn0_indices:
                                            #if matrix.loc['nots'][syn0_indice-max_distance:syn0_indice].sum() == 0: # PARA AL MENOS UN SYN0 NO HAY NOTS PROXIMOS, POR LO QUE ESTE OR_GROUP SATISFACE
                                            primera_posicion = 0 if syn0_indice-max_distance<0 else syn0_indice-max_distance
                                            if sum(nots_array[primera_posicion:syn0_indice]) == 0: # PARA AL MENOS UN SYN0 NO HAY NOTS PROXIMOS, POR LO QUE ESTE OR_GROUP SATISFACE
                                                ors_compliance = True
                                elif len(syn0_indices) == 0 and len(syn1_indices) > 0: #HAY SYN1 PERO NO HAY SYN0
                                    min_posicion = max(min(syn1_indices)-max_distance, 0)
                                    max_posicion = min(max(syn1_indices)+max_distance, len(opinion_text))
                                    tokens_implicados = opinion_text[min_posicion:max_posicion+1]
                                    
                                    syn2_words = or_group['syn2']
                                    syn2_indices = []
                                    for syn2_word in syn2_words:
                                        syn2_indices.extend([i for i, j in enumerate(tokens_implicados) if j == syn2_word])
                                    if len(syn2_indices) > 0 or len(syn2_words) == 0: # HAY SYN2, POR LO QUE EL OR_GROUP PUEDE SATISFACER
                                        syn2_array = [0] * len(tokens_implicados)
                                        for syn2_indice in syn2_indices:
                                            syn2_array[syn2_indice] = 1
                    
                                        syn1_indices = [x-min_posicion for x in syn1_indices]
                                        syn1_indices_sobreviven = []
                                        for indice in syn1_indices:
                                            if sum(syn2_array[indice-max_distance:indice+max_distance+1]) > 0:
                                                syn1_indices_sobreviven.append(indice)
                                        
                                        syn1_array_sobreviven = [0] * len(tokens_implicados)
                                        for syn1_indice in syn1_indices_sobreviven:
                                            syn1_array_sobreviven[syn1_indice] = 1
                    
                                        if len(syn1_array_sobreviven) > 0: #SOBREVIVE ALGUN SYN1, POR LO QUE EL OR_GROUP PUEDE SATISFACER
                                            # CONSTRUCCION ARRAY DE NOTS
                                            nots_words = wordset_ors[0]['nots']
                                            nots_indices = []
                                            for nots_word in nots_words:
                                                nots_indices.extend([i for i, j in enumerate(tokens_implicados) if j == nots_word])
                                            if len(nots_indices) == 0: # NO HAY NOTS, POR LO QUE EL OR_GROUP SATISFACE Y LA OPINION SE ACEPTA
                                                ors_compliance = True
                                                break
                                            else:
                                                nots_array = [0] * len(tokens_implicados)
                                                for nots_indice in nots_indices:
                                                    nots_array[nots_indice] = 1
                                                # CONSTRUCCION MATRIZ
                                                #matrix = pd.DataFrame(zip(syn1_array_sobreviven, nots_array)).transpose()
                                                #matrix.columns = tokens_implicados
                                                #matrix.index = ['syn1', 'nots']
                                                for syn1_indice in syn1_indices_sobreviven:
                                                    primera_posicion = 0 if syn1_indice-max_distance<0 else syn1_indice-max_distance
                                                    if sum(nots_array[primera_posicion:syn1_indice]) == 0: # PARA AL MENOS UN SYN1 NO HAY NOTS PROXIMOS, POR LO QUE ESTE OR_GROUP SATISFACE
                                                    #if matrix.loc['nots'][syn1_indice-max_distance:syn1_indice].sum() == 0: # PARA AL MENOS UN SYN1 NO HAY NOTS PROXIMOS, POR LO QUE ESTE OR_GROUP SATISFACE
                                                        ors_compliance = True
                                elif len(syn0_indices) > 0 and len(syn1_indices) > 0: #HAY SYN1 Y SYN0
                                    posiciones_syn = []
                                    posiciones_syn.extend(syn0_indices)
                                    posiciones_syn.extend(syn1_indices)
                                    min_posicion = max(min(posiciones_syn)-max_distance, 0)
                                    max_posicion = min(max(posiciones_syn)+max_distance, len(opinion_text))
                                    tokens_implicados = opinion_text[min_posicion:max_posicion+1]
                                    syn0_array = [0] * len(tokens_implicados)
                                    syn0_indices = [x-min_posicion for x in syn0_indices]
                                    for syn0_indice in syn0_indices:
                                        syn0_array[syn0_indice] = 1
                                    # CONSTRUCCION ARRAY DE NOTS
                                    nots_words = wordset_ors[0]['nots']
                                    nots_indices = []
                                    for nots_word in nots_words:
                                        nots_indices.extend([i for i, j in enumerate(tokens_implicados) if j == nots_word])
                                    if len(nots_indices) == 0: # NO HAY NOTS, POR LO QUE EL OR_GROUP SATISFACE Y LA OPINION SE ACEPTA
                                        ors_compliance = True
                                        break
                                    else:
                                        nots_array = [0] * len(tokens_implicados)
                                        for nots_indice in nots_indices:
                                            nots_array[nots_indice] = 1
                                        # CONSTRUCCION MATRIZ
                                        #matrix = pd.DataFrame(zip(syn0_array, nots_array)).transpose()
                                        #matrix.columns = tokens_implicados
                                        #matrix.index = ['syn0', 'nots']
                                        
                                        for syn0_indice in syn0_indices:
                                            primera_posicion = 0 if syn0_indice-max_distance<0 else syn0_indice-max_distance
                                            if sum(nots_array[primera_posicion:syn0_indice]) == 0: # PARA AL MENOS UN SYN0 NO HAY NOTS PROXIMOS, POR LO QUE ESTE OR_GROUP SATISFACE
                                            #if matrix.loc['nots'][syn0_indice-max_distance:syn0_indice].sum() == 0: # PARA AL MENOS UN SYN0 NO HAY NOTS PROXIMOS, POR LO QUE ESTE OR_GROUP SATISFACE
                                                ors_compliance = True
                                        if not ors_compliance: #LOS SYN0 NO SATISFACEN, HAY QUE VER LOS SYN1
                                            syn2_words = or_group['syn2']
                                            syn2_indices = []
                                            for syn2_word in syn2_words:
                                                syn2_indices.extend([i for i, j in enumerate(tokens_implicados) if j == syn2_word])
                                            if len(syn2_indices) > 0 or len(syn2_words) == 0: # HAY SYN2, POR LO QUE EL OR_GROUP PUEDE SATISFACER
                                                syn2_array = [0] * len(tokens_implicados)
                                                for syn2_indice in syn2_indices:
                                                    syn2_array[syn2_indice] = 1
                            
                                                syn1_indices = [x-min_posicion for x in syn1_indices]
                                                syn1_indices_sobreviven = []
                                                for indice in syn1_indices:
                                                    if sum(syn2_array[indice-max_distance:indice+max_distance+1]) > 0:
                                                        syn1_indices_sobreviven.append(indice)
                                                
                                                syn1_array_sobreviven = [0] * len(tokens_implicados)
                                                for syn1_indice in syn1_indices_sobreviven:
                                                    syn1_array_sobreviven[syn1_indice] = 1
                            
                                                if len(syn1_array_sobreviven) > 0: #SOBREVIVE ALGUN SYN1, POR LO QUE EL OR_GROUP PUEDE SATISFACER
                                                    # CONSTRUCCION ARRAY DE NOTS
                                                    nots_words = wordset_ors[0]['nots']
                                                    nots_indices = []
                                                    for nots_word in nots_words:
                                                        nots_indices.extend([i for i, j in enumerate(tokens_implicados) if j == nots_word])
                                                    if len(nots_indices) == 0: # NO HAY NOTS, POR LO QUE EL OR_GROUP SATISFACE Y LA OPINION SE ACEPTA
                                                        ors_compliance = True
                                                        break
                                                    else:
                                                        nots_array = [0] * len(tokens_implicados)
                                                        for nots_indice in nots_indices:
                                                            nots_array[nots_indice] = 1
                                                        # CONSTRUCCION MATRIZ
                                                        #matrix = pd.DataFrame(zip(syn1_array_sobreviven, nots_array)).transpose()
                                                        #matrix.columns = tokens_implicados
                                                        #matrix.index = ['syn1', 'nots']
                                                        for syn1_indice in syn1_indices_sobreviven:
                                                            primera_posicion = 0 if syn1_indice-max_distance<0 else syn1_indice-max_distance
                                                            if sum(nots_array[primera_posicion:syn1_indice]) == 0: # PARA AL MENOS UN SYN1 NO HAY NOTS PROXIMOS, POR LO QUE ESTE OR_GROUP SATISFACE
                                                            #if matrix.loc['nots'][syn1_indice-max_distance:syn1_indice].sum() == 0: # PARA AL MENOS UN SYN1 NO HAY NOTS PROXIMOS, POR LO QUE ESTE OR_GROUP SATISFACE
                                                                ors_compliance = True
        if and_compliance and (ors_compliance or elementos_or == 0):
            rows.append({'doc_number': opinion[1]['doc_number']})
            j += 1
    ##################################################_C_O_R_E_########################################
        if i % 200000 == 0:
            timestamps = calcula_y_muestra_tiempos('BUCLE OPINIONES: '+str(i)+' DE '+str(len(df))+' HITS: '+str(j), timestamps)
        i += 1
    #timestamps = calcula_y_muestra_tiempos('FINALIZA EL BUCLE DE OPINIONES', timestamps)
    rows = pd.DataFrame(rows)
    
    if len(rows) == 0:
        df_filtered = df.iloc[0:0]
        resultados_agregados = df[['rating', 'doc_number']].iloc[0:0]
        calcula_y_muestra_tiempos('FIN FUNCIÓN ANALIZE_WORDSET', timestamps)
        print('No se ha hallado')
    else:
        df_filtered = df.loc[rows['doc_number']]
        resultados_agregados = df_filtered.groupby('rating').agg({'doc_number': np.size})
        resultados_agregados_index = resultados_agregados.index
        resultados_agregados = resultados_agregados.to_dict(orient='index')
        
        if 1 not in resultados_agregados_index:
            resultados_agregados['1'] = {'doc_number':0}
        if 2 not in resultados_agregados_index:
            resultados_agregados['2'] = {'doc_number':0}
        if 3 not in resultados_agregados_index:
            resultados_agregados['3'] = {'doc_number':0}
        if 4 not in resultados_agregados_index:
            resultados_agregados['4'] = {'doc_number':0}
        if 5 not in resultados_agregados_index:
            resultados_agregados['5'] = {'doc_number':0}
            
        resultados_agregados = pd.DataFrame(resultados_agregados).transpose()

        calcula_y_muestra_tiempos('FIN FUNCIÓN ANALIZE_WORDSET', timestamps)
    
        if show:
            print_simple_histogram(resultados_agregados['doc_number'], title = "Analize para el wordset '" + wordset_name + "'")
    
    return df_filtered, resultados_agregados
########################################################################################################################
def get_ws_or_words(wsw):
    wordset_ors = wsw['wordset']['ors']
    words_list = []
    for or_group in wordset_ors:
        for syn0_word in or_group['syn0']:
            words_list.append(syn0_word)
        for syn1_word in or_group['syn1']:
            words_list.append(syn1_word)
    return words_list
########################################################################################################################
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load('en', disable=['parser', 'ner'])
    #"""https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
########################################################################################################################
def execute_preprocessing_pipeline(opinions_raw, eliminar_poco_frecuentes=True, lemmatize=False):

    timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN EXECUTE_PREPROCESSING_PIPELINE', timestamps=[])
    categorias_gramaticales_admitidas = ['NOUN', 'ADJ', 'VERB', 'ADV']
    summaries_raw = opinions_raw.summary.values.tolist()
    summaries_words = sent_to_words(summaries_raw)
    
    
    num_docs = opinions_raw.index.values

    i = 0
    for i in range(0, len(summaries_words)):
        #summaries_words[i] = ['not' if x=='didn' or x =='didnt' or x =='doesn' or x=='doesnt' or x=='dont' or x=='don' else x for x in summaries_words[i]]
        summaries_words[i] = mapear_palabras_especiales(summaries_words[i])
        i += 1
    summaries_words_nostop = remove_stopwords(summaries_words)
    summaries_bigrams = make_bigrams(summaries_words_nostop, summaries_words)

    timestamps = calcula_y_muestra_tiempos('SUMMARIES PROCESADOS', timestamps)
    reviews_raw = opinions_raw.review.values.tolist()
    reviews_words = sent_to_words(reviews_raw)
    i = 0
    for i in range(0, len(reviews_words)):
        #reviews_words[i] = ['not' if x=='didn' or x =='didnt' or x =='doesn' or x=='doesnt' or x=='dont' or x=='don' else x for x in reviews_words[i]]
        reviews_words[i] = mapear_palabras_especiales(reviews_words[i])
        i += 1
    reviews_words_nostop = remove_stopwords(reviews_words)
    reviews_bigrams = make_bigrams(reviews_words_nostop, reviews_words)
    timestamps = calcula_y_muestra_tiempos('REVIEWS PROCESADAS', timestamps)
    
    if lemmatize:
        reviews_lemmatized = lemmatization(reviews_bigrams, allowed_postags=categorias_gramaticales_admitidas)
        summaries_lemmatized = lemmatization(summaries_bigrams, allowed_postags=categorias_gramaticales_admitidas)
        odf_false = build_odf(num_docs, opinions_raw.rating, summaries_lemmatized, summaries_raw, reviews_lemmatized, reviews_raw, None)
    else: 
        odf_false = build_odf(num_docs, opinions_raw.rating, summaries_bigrams, summaries_raw, reviews_bigrams, reviews_raw, None)
        
    timestamps = calcula_y_muestra_tiempos('ODF_FALSE CONSTRUIDO', timestamps)
    url_false = r'..\data\odfs\odf_false_'+str(len(odf_false))+'.csv'
    odf_false.to_csv (url_false, index = False, header=True)
    timestamps = calcula_y_muestra_tiempos('ODF_FALSE SALVADO EN DATA/ODFS', timestamps)
    
    '''
    if eliminar_poco_frecuentes:
        odf_true = build_odf(num_docs, opinions_raw.rating, summaries_bigrams, summaries_raw, reviews_bigrams, reviews_raw, generar_corpus_aceptado(odf_false))
        timestamps = calcula_y_muestra_tiempos('ODF CONSTRUIDO DE NUEVO, ELIMINANDO PALABRAS POCO FRECUENTES', timestamps)
        url = r'..\data\odfs\odf_'+str(len(odf_true))+'.csv'
        
        odf_true.to_csv (url, index = False, header=True)
        timestamps = calcula_y_muestra_tiempos('ODF_TRUE SALVADO EN DATA/ODFS', timestamps)
        timestamps = calcula_y_muestra_tiempos('FIN FUNCIÓN EXECUTE_PREPROCESSING_PIPELINE', timestamps)
        
        return odf_false, odf_true'''

    calcula_y_muestra_tiempos('FIN FUNCIÓN EXECUTE_PREPROCESSING_PIPELINE', timestamps)
    
    return odf_false
########################################################################################################################
def generate_bow(df, classify_pos_b=True, show=False):
    timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN GENERATE_BOW', timestamps=[])
    bigrams = []
    
    i = 0
    for opinion in df.iterrows():
        text_list = opinion[1]['text']
        if text_list.count(', ') > 0:
            bigrams.extend(text_list.split(', '))
        i += 1
        if i%200000 == 0:
            timestamps = calcula_y_muestra_tiempos('BUCLE GENERANDO BOW: i='+str(i)+' de '+str(len(df)), timestamps)
    timestamps = calcula_y_muestra_tiempos('LISTA DE BIGRAMS FINALIZADA', timestamps)
    aux = pd.Series(bigrams).value_counts()
    
    bag = pd.DataFrame(zip(aux.index, aux.values), columns=['bigram', 'num_occurrences']).sort_values('num_occurrences', ascending=False)
    #if classify_pos_b:
    #    bag['pos'] = bag.apply(lambda row: classify_pos(row['bigram']), axis=1)
    if show:
        print_wordcloud_from_frequencies(bag[['bigram', 'num_occurrences']], 18)
        
    timestamps = calcula_y_muestra_tiempos('FIN FUNCIÓN GENERATE_BOW', timestamps)
    return bag
########################################################################################################################
def print_wordcloud_from_frequencies(bag, max_words=999):
    import matplotlib.pyplot as plt
    d = {}
    for a, x in bag.values:
        d[a] = x   
    wordcloud = WordCloud(max_words = max_words)
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
########################################################################################################################
'''
def generar_corpus_aceptado(odf):
    bow=generate_bow(odf, False)
    bow['num_occurrences'][0]
    corpus_aceptado = bow[bow['num_occurrences'] > 2]
    return corpus_aceptado'''
########################################################################################################################
'''def extract_bigrams_from_bow_(bow):
    bigrams = []
    number = []
    for item in bow.iterrows():
        if '_' in item[1]['bigram']:
            bigrams.append(item[1]['bigram'])
            number.append(item[1]['num_occurrences'])
    retorno = pd.DataFrame(list(zip(bigrams, number)), columns =['bigram', 'num_occurrences'])
    
    return retorno'''
########################################################################################################################
def extract_bigrams_from_counter(counter):
    bigrams = []
    number = []
    for token in counter:
        if '_' in token:
            bigrams.append(token)
            number.append(counter[token])
    retorno = pd.DataFrame(list(zip(bigrams, number)), columns =['bigram', 'num_occurrences'])
    retorno.sort_values(by ='num_occurrences', ascending=False)
    
    return retorno
########################################################################################################################
def load_latest_odf(nrows, is_false=False):
    import os
    dirname = os.getcwd()
    if is_false:
        url = os.path.join(dirname, r'../data/odfs/odf_false_'+str(nrows)+'.csv')    
    else:
        url = os.path.join(dirname, r'../data/odfs/odf_'+str(nrows)+'.csv')
    try:
        with open(url):
            df = pd.read_csv(url, keep_default_na=False)
            df.index = df['doc_number']
            return df
    except IOError:
        print("El fichero no existe")
    return None
########################################################################################################################
def remove_stopwords_from_bow_2(bow, stopwords_modo):
    bow2 = bow.copy()
    bow2['token'] = bow2.index.values
    rows_to_delete = []
    i = 0
    for i in range(0, len(bow2)):
        if bow2['token'][i] in get_stopwords(modo=stopwords_modo):
            #rows_to_delete.append(i)
            rows_to_delete.append(bow2['token'][i])
    
    return rows_to_delete
########################################################################################################################
def analize_wordset_occurrences(df, lista_resultados_analize):
    timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN ANALIZE_WORDSET_OCCURRENCES', timestamps=[])
    matriz_documento_wordset = pd.DataFrame(index=df.index)
    for resultado_busqueda in lista_resultados_analize:
        nombre = resultado_busqueda['name']
        timestamps = calcula_y_muestra_tiempos('SE PROCEDE A ANALIZAR EL WORDSET ' + nombre, timestamps=timestamps)
        matriz_documento_wordset[nombre] = 0
        i = 0
        for doc_number in resultado_busqueda['resultados']['doc_number']:
            matriz_documento_wordset[nombre][doc_number] = 1
            if i%20000 == 0:
                timestamps = calcula_y_muestra_tiempos('BUCLE: '+str(i)+' de '+str(len(resultado_busqueda['resultados']['doc_number'])), timestamps)
            i += 1
    matriz_documento_wordset['total_wordsets'] = matriz_documento_wordset.sum(axis=1)
    matriz_documento_wordset_agg = matriz_documento_wordset.groupby('total_wordsets').count()[matriz_documento_wordset.groupby('total_wordsets').count().columns[0]].to_frame()
    matriz_documento_wordset_agg.columns.values[0] = 'total_documents'
    print_simple_histogram(matriz_documento_wordset_agg['total_documents'], title = '# de opiniones por # de hits')
    print ('La ocupación parcial es del', "{:.0%}".format(np.count_nonzero(matriz_documento_wordset['total_wordsets'])/len(matriz_documento_wordset)))
    print ('La ocupación total es del', "{:.0%}".format(matriz_documento_wordset.sum().sum()/(matriz_documento_wordset.size-len(matriz_documento_wordset))))
    
    mat_doc_ws_expanded = pd.merge(matriz_documento_wordset, df['rating'], left_index=True, right_index=True, how='inner')
    wordsets_names = list(matriz_documento_wordset.columns)
    wordsets_names.remove('total_wordsets')
    mat_doc_ws_expanded_agg = mat_doc_ws_expanded.groupby(wordsets_names).agg({'total_wordsets':[np.size, np.mean], 'rating':[np.median, np.mean, np.std]})
    mat_doc_ws_expanded_agg.columns = ['total_opiniones', 'total_temas', 'rating_median', 'rating_mean', 'rating_sd']
    calcula_y_muestra_tiempos('FIN FUNCIÓN ANALIZE_WORDSET_OCCURRENCES', timestamps=timestamps)

    return mat_doc_ws_expanded, mat_doc_ws_expanded_agg
########################################################################################################################
def get_stopwords(modo='pre'):
    stop_words = stopwords.words('english')
    stop_words.extend(['one', 'get', 'got', 'getting', 'shoud', 'like', 'also', 'would', 'even', 'could', 'two', 'item', 'thing', 'put', 'however',
                       'something', 'etc', 'unless', 'https', 'www', 'for'])
    if modo=='pre':
        palabras_deseadas = ['too', 'not', 'no', 'nor', 'than', 'out']
        for palabra in palabras_deseadas:
            stop_words.remove(palabra)
    return stop_words
########################################################################################################################
def mapear_palabras_especiales(array_input):
    retorno = ['not'    if x == 'didn' 
                        or x == 'didnt' 
                        or x == 'dint'
                        or x == 'didnot'
                        or x == 'dident'
                        or x == 'doesn' 
                        or x == 'doesnt'
                        or x == 'dosn'
                        or x == 'doens'
                        or x == 'dosen'
                        or x == 'dosnt'
                        or x == 'doesnot'
                        or x == 'dosent'
                        or x == 'dont' 
                        or x == 'don' 
                        or x == 'couldn'
                        or x == 'couldnt'
                        or x == 'wouldn'
                        or x == 'wouldnt'
                        or x == 'cant'
                        else x for x in array_input]
    return retorno
########################################################################################################################
def get_wsw_structure(wsw):
    wordset_ands = wsw['wordset']['ands']
    wordset_ors = wsw['wordset']['ors']
        
    elementos_or = 0
    for or_group in wordset_ors:
        elementos_or += len(or_group['syn0'])+len(or_group['syn1'])+len(or_group['syn2'])+len(or_group['nots'])

    return len(wordset_ands), elementos_or
########################################################################################################################
def get_close_words(df, word, max_distance=3, n_words=8):
    timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN GET_CLOSE_WORDS', timestamps=[])
    palabras_antes = []
    palabras_despues  = []
    ratings_antes = []
    ratings_despues = []
    
    i = 0
    for opinion in df.iterrows():
        texto = opinion[1]['text'].split(', ')
        if word in set(texto):
            word_indices = []
            word_indices.extend([i for i, j in enumerate(texto) if j == word])
            for indice in word_indices:
                texto_despues = texto[indice+1:indice+max_distance+1]
                palabras_despues.extend(texto_despues)
                ratings_despues.extend(len(texto_despues)*[opinion[1]['rating']])
                texto_antes = texto[indice-max_distance:indice]
                palabras_antes.extend(texto_antes)
                ratings_antes.extend(len(texto_antes)*[opinion[1]['rating']])
        if i%200000 == 0:
            timestamps = calcula_y_muestra_tiempos('BUCLE OPINIONES: '+str(i)+' DE '+str(len(df)), timestamps=timestamps)
        i += 1

    df_antes = pd.DataFrame(list(zip(palabras_antes, ratings_antes)), columns = ['token', 'rating'])
    df_antes = df_antes.groupby('token').agg({'rating':[np.size, np.mean, np.std]}).fillna(0)
    df_antes = df_antes.drop(remove_stopwords_from_bow_2(df_antes, 'post'))
    df_antes.columns = ['num_occurrences', 'rating_mean', 'rating_sd']
    df_antes = df_antes.sort_values(by=['num_occurrences'], ascending=False)

    df_despues = pd.DataFrame(list(zip(palabras_despues, ratings_despues)), columns = ['token', 'rating'])
    df_despues = df_despues.groupby('token').agg({'rating':[np.size, np.mean, np.std]}).fillna(0)
    df_despues = df_despues.drop(remove_stopwords_from_bow_2(df_despues, 'post'))
    df_despues.columns = ['num_occurrences', 'rating_mean', 'rating_sd']
    df_despues = df_despues.sort_values(by=['num_occurrences'], ascending=False)
    
    calcula_y_muestra_tiempos('FIN FUNCIÓN GET_CLOSE_WORDS', timestamps=timestamps)
    
    return df_antes.iloc[:n_words], df_despues.iloc[:n_words]
########################################################################################################################  
def get_network_color_map():
    return {0:'#E6E6E6', 1:'#cc3232', 2:'#db7b2b', 3:'#e7b416', 4:'#99c140', 5:'#2dc937'}
########################################################################################################################
def visualize_wordsets_network_6(matriz_doc_ws_expanded, group_size=200, k=0.1, ratings='F', mostrar_opiniones = False):
    timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN VISUALIZE_WORDSETS_NETWORK', timestamps=[])
    #matriz_doc_ws_expanded = mat_doc_ws_expanded.copy()
    #ratings='F'
    #k=0.1
    #group_size=250
    
    wordsets_names_extended = get_extended_wordsets_names(matriz_doc_ws_expanded)
    wordsets_names = get_wordsets_names(matriz_doc_ws_expanded)
    
    if ratings != 'F':
        matriz_doc_ws_expanded = matriz_doc_ws_expanded[matriz_doc_ws_expanded['rating'].isin(ratings)]
    
    mat = matriz_doc_ws_expanded.groupby(wordsets_names_extended).agg({'total_wordsets':np.size})
    mat.columns =(['num_opiniones'])
    mat['num_opiniones_mod'] = round(mat['num_opiniones']/group_size)
    mat = mat[mat['num_opiniones_mod'] >= 1]
    
    num_wordsets = len(wordsets_names)
    timestamps = calcula_y_muestra_tiempos('SE PROCEDE A GENERAR LA EDGELIST', timestamps=timestamps)
    edgelist = []
    for item in mat.iterrows():
        index_as_list = list(item[0])
        rating = index_as_list.pop()
        node_id = random.randint(0, 999999999)
        i = 0
        for tema in index_as_list:
            if tema == 1:
                edge = {'from':node_id, 'to':wordsets_names[i], 'rating':rating, 'width':int(item[1]['num_opiniones_mod'])}
                edgelist.append(edge)
            i += 1
            
    edgelist = pd.DataFrame(edgelist)
    
    timestamps = calcula_y_muestra_tiempos('EDGELIST GENERADA. SE PROCEDE A GENERAR LA NODELIST', timestamps=timestamps)
    nodelist = wordsets_names.copy()
    nodelist.extend(edgelist['from'].unique())
    nodelist = pd.DataFrame(nodelist, columns=['node_name'])
    
    nodelist = pd.merge(nodelist, edgelist.groupby('from').agg({'rating':np.mean})['rating'], left_on='node_name', right_on='from', how='outer')
    nodelist.fillna(0, inplace=True)
    
    if mostrar_opiniones:
        nodelist = pd.merge(nodelist, pd.concat([edgelist.groupby('to').agg({'width':np.sum}), edgelist.groupby('from').agg({'width':np.mean})])['width']*10, left_on='node_name', right_index=True, how='outer')
        nodelist.rename(columns={'width':'size'}, inplace=True)
    else:    
        sizes = matriz_doc_ws_expanded[wordsets_names].sum().tolist()
        ct = 2000/max(sizes)
        sizes = [size * ct for size in sizes]
        sizes.extend((len(nodelist)-num_wordsets)*[0])
        nodelist['size'] = sizes
    
    G = nx.Graph()
        
    timestamps = calcula_y_muestra_tiempos('SE PROCEDE A POBLAR NODOS Y ARISTAS', timestamps=timestamps)
    # NODES
    for index, row in nodelist.iterrows():
        G.add_node(row['node_name'], group=row['rating'], color=get_network_color_map()[row['rating']], size=row['size'])
    
    # EDGES
    for index, row in edgelist.iterrows():
        G.add_edge(row['from'], row['to'], color=get_network_color_map()[row['rating']], width=row['width'])
    
    labels = {}
    for node in nodelist['node_name'][:num_wordsets]:
        labels[node] = node
     
    plt.figure(figsize=(25, 25))
    timestamps = calcula_y_muestra_tiempos('SE PROCEDE A CALCULAR POS', timestamps=timestamps)
    pos = nx.spring_layout(G, k=k, iterations=100)
    edges, edge_colors = zip(*nx.get_edge_attributes(G, 'color').items())
    edges, edge_widths = zip(*nx.get_edge_attributes(G, 'width').items())
    nodes, node_colors = zip(*nx.get_node_attributes(G, 'color').items())
    nodes, node_sizes = zip(*nx.get_node_attributes(G, 'size').items())
    
    timestamps = calcula_y_muestra_tiempos('SE PROCEDE A DIBUJAR LA RED', timestamps=timestamps)
    nx.draw(G, pos=pos, nodelist=nodes, node_size=node_sizes, node_color=node_colors, edgelist=edges, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_labels(G, pos, labels, font_size=20)
    plt.savefig('../viz/'+datetime.now().strftime('%Y%m%d%H%M%S')+'.jpg')
    plt.show()
    calcula_y_muestra_tiempos('FIN DE LA FUNCION VISUALIZE_WORDSETS_NETWORK', timestamps=timestamps)
########################################################################################################################
def busca_tokens(counter, words):
    #timestamps = calcula_y_muestra_tiempos('INICIO FUNCIÓN BUSCA_TOKENS', timestamps=[])
    tokens_a_retornar = []
    for word in words:
        for elemento in set(counter): # como set el rendimiento mejora espectacularmente
            if word == elemento or elemento.split('_').count(word):
                tokens_a_retornar.append(elemento)
    #timestamps = calcula_y_muestra_tiempos('FIN FUNCIÓN BUSCA_TOKENS', timestamps)
    return tokens_a_retornar
########################################################################################################################
def get_heatmap_cmap(grey_shades=True):
    if grey_shades:
        return sns.light_palette("#000000", as_cmap=True)
    else:
        return sns.diverging_palette(10, 150, sep=20, as_cmap=True)
########################################################################################################################
def get_wordsets_names(matriz_doc_ws):
    wordsets_names = list(matriz_doc_ws.columns)
    wordsets_names.remove('total_wordsets')
    wordsets_names.remove('rating')
    return wordsets_names
########################################################################################################################
def get_extended_wordsets_names(mat_doc_ws):
    wordsets_names_extended = get_wordsets_names(mat_doc_ws)
    wordsets_names_extended.append('rating')
    return wordsets_names_extended
########################################################################################################################
def get_reduced_mat_doc_ws(mat_doc_ws):
    return mat_doc_ws.drop(['total_wordsets', 'rating'], axis=1)
########################################################################################################################
def get_sharing_matrix(mat_doc_ws):
    wordsets_names = get_wordsets_names(mat_doc_ws)
    mat_doc_ws_reduced = get_reduced_mat_doc_ws(mat_doc_ws)
    df = pd.DataFrame([np.zeros(len(wordsets_names)) for i in range(0, len(wordsets_names))], index=wordsets_names, columns=wordsets_names)
    
    for tema in wordsets_names:
        df[tema] = mat_doc_ws_reduced[mat_doc_ws_reduced[tema]==1].sum()/mat_doc_ws_reduced.sum()
    
    df2_1 = df.copy()
    for i in range(0, len(wordsets_names)):
        df2_1.iloc[i][i] = 0
        
    wordsets_names = get_wordsets_names(mat_doc_ws)
    mat_doc_ws_reduced = get_reduced_mat_doc_ws(mat_doc_ws)
    df = pd.DataFrame([np.zeros(len(wordsets_names)) for i in range(0, len(wordsets_names))], index=wordsets_names, columns=wordsets_names)
    
    for tema in wordsets_names:
        df.loc[tema] = mat_doc_ws_reduced[mat_doc_ws_reduced[tema]==1].sum()/mat_doc_ws_reduced.sum()
    
    df2_2 = df.copy()
    for i in range(0, len(wordsets_names)):
        df2_2.iloc[i][i] = 0
  
    df3 = pd.DataFrame([np.zeros(len(wordsets_names)) for i in range(0, len(wordsets_names))], index=wordsets_names, columns=wordsets_names)
    
    df3 = (df2_1+df2_2)/2
    
    for i in range(0, len(wordsets_names)):
        for j in range(i, len(wordsets_names)):
            df3.iloc[i][j] = 0 #df.iloc[j][i]
    
    sns.heatmap(df3, cmap=get_heatmap_cmap())

    return df3
########################################################################################################################
def calculate_heatmap_matrix(mat_doc_ws):
    wordsets_names = get_wordsets_names(mat_doc_ws)
    heatmap = mat_doc_ws.groupby('rating').sum()[wordsets_names]
    heatmap.index.names = ['Puntuación']
    return heatmap
########################################################################################################################
def print_topic_heatmaps(mat_doc_ws):
    heatmap = calculate_heatmap_matrix(mat_doc_ws)
    heatmap_n = heatmap.div(heatmap.max(axis=0), axis=1) # LEER EN HORIZONTAL, ES DECIR CADA TEMA
    sns.heatmap(heatmap_n.transpose(), cmap=get_heatmap_cmap())
########################################################################################################################
def print_rating_heatmaps(mat_doc_ws):
    heatmap = calculate_heatmap_matrix(mat_doc_ws)
    heatmap_n2 = heatmap.div(heatmap.max(axis=1), axis=0)/get_reduced_mat_doc_ws(mat_doc_ws).sum()
    ct = 1/max(heatmap_n2.max())
    heatmap_n2 = heatmap_n2 * ct
    #sns.heatmap(heatmap_n2.transpose(), cmap=get_heatmap_cmap()) # LEER EN VERTICAL, ES DECIR CADA RATING
    
    heatmap_n3 = heatmap_n2.div(heatmap_n2.max(axis=1), axis=0)
    #sns.heatmap(heatmap_n2.transpose(), cmap=get_heatmap_cmap()) # LEER EN VERTICAL, ES DECIR CADA RATING
    sns.heatmap(heatmap_n3.transpose(), cmap=get_heatmap_cmap()) # LEER EN VERTICAL, ES DECIR CADA RATING
        
    
########################################################################################################################
def get_popular_topic_combinations(mat_doc_ws_agg, wordsets_names, n_temas=10):
    lista_combinaciones_populares = []
    for row in mat_doc_ws_agg[mat_doc_ws_agg['total_temas'] > 1].iterrows():
        combinacion = []
        combinacion.append(row[1]['total_opiniones'])
        index_as_list = list(row[0])
        i = 0
        for tema in index_as_list:
            if tema == 1:
                combinacion.append(wordsets_names[i])
            i += 1
        lista_combinaciones_populares.append(combinacion)
    
    df_t = pd.DataFrame(lista_combinaciones_populares).sort_values(by=0, ascending=False)
    df_t.rename(columns={0:'Núm. de hits'}, inplace=True)
    #df_t.drop([3,4,5,6,7,8,9,10,11,12], axis=1, inplace=True)
    df_t = df_t[['Núm. de hits', 1, 2]]
    return df_t.iloc[:n_temas]
########################################################################################################################
def statistics_by_topic(mat_doc_ws):
    dic = []
    for tema in get_wordsets_names(mat_doc_ws):
        mat = mat_doc_ws[get_extended_wordsets_names(mat_doc_ws)].groupby(tema).agg({'rating':[np.size, np.mean, np.std]}).transpose()
        mat['0_size'] = int(mat.iloc[0][0])
        mat['0_mean'] = mat.iloc[1][0]
        mat['0_sd'] = mat.iloc[2][0]
        mat['1_size'] = int(mat.iloc[0][1])
        mat['1_mean'] = mat.iloc[1][1]
        mat['1_sd'] = mat.iloc[2][1]
        mat.drop([0, 1], axis=1, inplace=True)
        mat.drop(mat.index[1], inplace=True)
        mat.drop(mat.index[1], inplace=True)
        mat.set_index(pd.Series([tema]), inplace=True)
        mat = {'Tema':tema, '1_size':mat['1_size'][0], '1_mean':mat['1_mean'][0], '1_sd':mat['1_sd'][0], '0_size':mat['0_size'][0], '0_mean':mat['0_mean'][0], '0_sd':mat['0_sd'][0]}
        dic.append(mat)
    df_temas = pd.DataFrame.from_dict(dic)
    return df_temas
########################################################################################################################
def print_statistics_by_topic_heatmap_rating_value(mat_doc_ws):
    heatmap = statistics_by_topic(mat_doc_ws)
    heatmap.index=heatmap['Tema']
    heatmap.drop(['Tema', '0_size', '1_size', '0_sd', '1_sd'], axis=1, inplace=True)
    heatmap = heatmap.rename(columns={'0_mean':'No Hit', '1_mean':'Hit'})
    sns.heatmap(heatmap, cmap=get_heatmap_cmap(False)) # LEER EN VERTICAL
########################################################################################################################
def print_statistics_by_topic_heatmap_rating_sd(mat_doc_ws):
    heatmap = statistics_by_topic(mat_doc_ws)
    heatmap.index=heatmap['Tema']
    heatmap.drop(['Tema', '0_size', '1_size', '0_mean', '1_mean'], axis=1, inplace=True)
    heatmap = heatmap.rename(columns={'0_sd':'No Hit', '1_sd':'Hit'})
    sns.heatmap(heatmap, cmap=get_heatmap_cmap()) # LEER EN VERTICAL
########################################################################################################################
def get_respuesta_comprador(lista_resultados_analize, matriz_doc_ws):
    lineas = []
    for each in lista_resultados_analize:
        lineas.append({'Tema':each['name'], 'slope':round(LinearRegression().fit(each['agregados'].index.values.reshape((-1, 1)), each['agregados']['doc_number']).coef_[0], 1)})
    respuesta_comprador = pd.DataFrame.from_dict(lineas)
    respuesta_comprador['Coef. Respuesta'] = round(respuesta_comprador['slope']/statistics_by_topic(matriz_doc_ws)['1_size'], 3)
    respuesta_comprador.sort_values(by='Coef. Respuesta', ascending=False, inplace=True)
    respuesta_comprador.drop(['slope'], axis=1, inplace=True)
    return respuesta_comprador
########################################################################################################################
def get_first_lines_electronics_5(n_rows):
    nrows = 10
    with open("../data/Electronics_5.json", "r") as f:
        counter = 0
        lines = []
     
        for line in f:
            line_dict = json.loads(line)
            if 'reviewText' in line_dict and 'overall' in line_dict and 'summary' in line_dict:
                lines.append(line_dict)
                counter += 1
            if counter == nrows: break

    return pd.DataFrame.from_dict(lines)
########################################################################################################################
def map_rating(rating_1_based):
    mapping = {1:0, 2:1, 3:2, 4:3, 5:4}
    if(isinstance(rating_1_based, (float, int, np.int64))):
        return mapping[rating_1_based]
    if(isinstance(rating_1_based, pd.Series)):
        return rating_1_based.map(mapping)
########################################################################################################################
def unmap_rating(rating_0_based):
    mapping = {0:1, 1:2, 2:3, 3:4, 4:5}
    if(isinstance(rating_0_based, (float, int, np.int64))):
        return mapping[rating_0_based]
    if(isinstance(rating_0_based, pd.Series)):
        return rating_0_based.map(mapping)
########################################################################################################################      
def evaluar_modelo_reg(modelo_entrenado, X_test, y_test):
   
    y_pred = modelo_entrenado.predict(X_test)
    y_diff = abs(y_pred-y_test)
    aggr = y_diff.groupby(y_diff).count()
    
    evaluar_modelo_core(aggr, y_test, y_pred)
########################################################################################################################
def evaluar_modelo_lstm(modelo_entrenado, dataloader):
   
    y_pred = []
    for x, y, l in dataloader:   
        y_pred.extend((torch.max(modelo_entrenado(x.long()), 1)[1]).tolist())
    y_diff = abs(pd.Series(y_pred)-pd.Series(dataloader.dataset.y))
    aggr = y_diff.groupby(y_diff).count()

    evaluar_modelo_core(aggr, dataloader.dataset.y, y_pred)
########################################################################################################################
def evaluar_modelo_core(aggr, y_true, y_pred):
    print_simple_histogram(aggr, title='Distribución de los errores')
    
    total_0_parc = aggr[0]
    total_1_parc = aggr[1]
    total_2_parc = aggr[2]
    total_3_parc = aggr[3]
    total_4_parc = aggr[4]
    
    total_0_acum = total_0_parc
    total_1_acum = total_0_acum+total_1_parc
    total_2_acum = total_1_acum+total_2_parc
    total_3_acum = total_2_acum+total_3_parc
    total_4_acum = total_3_acum+total_4_parc
    
    ratio_0_parc = total_0_parc/aggr.sum()
    ratio_1_parc = total_1_parc/aggr.sum()
    ratio_2_parc = total_2_parc/aggr.sum()
    ratio_3_parc = total_3_parc/aggr.sum()
    ratio_4_parc = total_4_parc/aggr.sum() 
    
    ratio_0_acum = total_0_parc/aggr.sum()
    ratio_1_acum = total_1_acum/aggr.sum()
    ratio_2_acum = total_2_acum/aggr.sum()
    ratio_3_acum = total_3_acum/aggr.sum()
    ratio_4_acum = total_4_acum/aggr.sum()
    
    str_0_acum = ("0: %.2f (%2d)" % (ratio_0_acum, total_0_acum)).ljust(20, ' ')
    str_0_parc = ("0: %.2f (%2d)" % (ratio_0_parc, total_0_parc)).ljust(20, ' ')
    str_1_acum = ("1: %.2f (%2d)" % (ratio_1_acum, total_1_acum)).ljust(20, ' ')
    str_1_parc = ("1: %.2f (%2d)" % (ratio_1_parc, total_1_parc)).ljust(20, ' ')
    str_2_acum = ("2: %.2f (%2d)" % (ratio_2_acum, total_2_acum)).ljust(20, ' ')
    str_2_parc = ("2: %.2f (%2d)" % (ratio_2_parc, total_2_parc)).ljust(20, ' ')
    str_3_acum = ("3: %.2f (%2d)" % (ratio_3_acum, total_3_acum)).ljust(20, ' ')
    str_3_parc = ("3: %.2f (%2d)" % (ratio_3_parc, total_3_parc)).ljust(20, ' ')
    str_4_acum = ("4: %.2f (%2d)" % (ratio_4_acum, total_4_acum)).ljust(20, ' ')
    str_4_parc = ("4: %.2f (%2d)" % (ratio_4_parc, total_4_parc)).ljust(20, ' ')
        
    print('Acumulados'.ljust(20, ' '), 'Parciales'.ljust(20, ' '))
    print(str_0_acum, str_0_parc)
    print(str_1_acum, str_1_parc)
    print(str_2_acum, str_2_parc)
    print(str_3_acum, str_3_parc)
    print(str_4_acum, str_4_parc)
    print()
    print("Accuracy: %.2f" % (ratio_0_acum))
    print("Acc1-: %.2f" % (ratio_1_acum))
    print("RMSE: %.2f" % (np.sqrt(mean_squared_error(y_pred, y_true))))
########################################################################################################################
def predict_from_text(modelo_entrenado, cv_entrenado, text):
    return unmap_rating(np.argmax(modelo_entrenado.predict_proba(cv_entrenado.transform([text]))))
########################################################################################################################
def get_all_topic_names(topics):
    names = []
    for topic_name in topics:
        names.append(topic_name)
    return names




















########################################################################################################################
def tokenize (text, tok):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]
########################################################################################################################
def encode_sentence_(text, vocab2index, N=70):
    #tokenized = tokenize(text, tok)
    tokenized = text.split(', ')
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length
########################################################################################################################
def encode_sentence(text, vocab2index, tok, N=70):
    tokenized = tokenize(text, tok)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length
########################################################################################################################
class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]
########################################################################################################################
'''def train_model(model, train_dl, val_dl, epochs=10, lr=0.001):
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
'''  
def train_model(model, train_dl, val_dl, epochs=10, lr=0.001):
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
        #print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))
        print("epoch %.2i: train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (i+1, sum_loss/total, val_loss, val_acc, val_rmse))    
########################################################################################################################
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
########################################################################################################################
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
########################################################################################################################
def build_token_counter(odf):
    token_counter = Counter()
    for index, row in odf.iterrows():
        token_counter.update(row['text'].split(', '))
    return token_counter