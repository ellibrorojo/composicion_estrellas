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
'''
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
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN GETDF')    
    if nrows == 9999999999:
        nrows = 6737497
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        if i % 50000 == 0:
            calcula_y_muestra_tiempos_2('BUCLE RAW DATA: i='+str(i)+' de '+str(nrows))
        if i == nrows:
            break
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN GETDF') 
    return pd.DataFrame.from_dict(df, orient='index')
########################################################################################################################
def electronics_5_to_raw_data_0(nrows=9999999999):
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN ELECTRONICS_5_TO_RAW_DATA_0')
    df = getDF(r'..\data\Electronics_5.json', nrows)
    calcula_y_muestra_tiempos_2('SE PROCEDE A CREAR EL CAMPO OPINION_YEAR')
    df['reviewTime'] = df.apply(lambda row: datetime.utcfromtimestamp(int(row['unixReviewTime'])).strftime('%Y%m%d'), axis=1)
    df['opinion_year'] = df.apply(lambda row: row['reviewTime'][:-4], axis=1)
    calcula_y_muestra_tiempos_2('SE PROCEDE A ELIMINAR COLUMNAS QUE NO USAREMOS')
    df = df.drop(columns=['vote', 'verified', 'style', 'reviewerName', 'unixReviewTime', 'image', 'reviewTime'])
    calcula_y_muestra_tiempos_2('SE PROCEDE A RENOMBRAR ALGUNAS COLUMNAS')
    df = df.rename(columns={'overall':'rating', 'reviewerID':'user_id', 'asin':'item_id', 'reviewText':'review'})
    calcula_y_muestra_tiempos_2('SE PROCEDE A SUSTITUIR LOS NANS POR CADENAS VACÍAS')
    df = df.fillna('')
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN ELECTRONICS_5_TO_RAW_DATA_0')
    return df
########################################################################################################################
def print_simple_histogram(data_agg, labels=None, title='Title'):
    #plt.ylabel('ylabel')
    #plt.xlabel('xlabel')

    '''
    x_values = data_agg.index.values
    fig, ax = plt.subplots()
    if len(x_values) > 5:
        fig.autofmt_xdate(rotation=45)
    #ax.set_xticklabels(x_values, rotation=45, ha='center')
    ax.bar(x_values, data_agg, color='blue')
    plt.show()'''
    
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
def print_simple_boxplot(data, labels, yscale, grid):
    fig, ax = plt.subplots()
    ax.set_yscale(yscale)
    if grid != 'None':
        ax.yaxis.grid(True, which = grid)
    ax.set_xticklabels(labels)
    ax.boxplot(data)
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
def calcula_y_muestra_tiempos_2(texto):
    timestamps.append(datetime.now())
    tot = len(timestamps)
    #↨print('::::', texto, '::::', strfdelta(timestamps[tot-1]-timestamps[tot-2]))
    if ['INICIO', 'FIN'].count(texto.split(' ')[0]) > 0:
        print((strfdelta(timestamps[tot-1]-timestamps[tot-2])+' ____ '+texto+' ').ljust(100, '_'))
    else:
        print((strfdelta(timestamps[tot-1]-timestamps[tot-2])+' :::: '+texto+' ').ljust(100, ':'))
########################################################################################################################

'''def show_aggregated_results(resultados_agregados):
    x_values = list(resultados_agregados.index.values)
    x_values = list(map(int, x_values))
    plt.ylabel('# Referencias')
    plt.xlabel('Rating')
    plt.title("# Menciones del elemento '" + "'")
    plt_summary = plt.bar(x_values, resultados_agregados['summary_length'], color='blue')
    plt_review = plt.bar(x_values, resultados_agregados['total_text']-resultados_agregados.summary_length, bottom=resultados_agregados['summary_length'], color='red')
    plt.legend((plt_summary[0], plt_review[0]), ('En summaries', 'En reviews'))
    plt.show()'''
########################################################################################################################
def print_stacked_histogram(data_agg, bottom_colname, bottom_legend, top_colname, top_legend):
    x_values = list(data_agg.index.values)
    x_values = list(map(int, x_values))
    #plt.ylabel('# Referencias')
    #plt.xlabel('Rating')
    #plt.title("# Menciones del elemento '" + "'")
    plt_bottom  = plt.bar(x_values, data_agg[bottom_colname], color='tab:blue')
    plt_top     = plt.bar(x_values, data_agg[top_colname]-data_agg.summary_length, bottom=data_agg[bottom_colname], color='tab:red')
    plt.legend((plt_bottom[0], plt_top[0]), (bottom_legend, top_legend))
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
        print_simple_histogram(raw_data.groupby('opinion_year').count()['item_id'])
    
    elif analysis == 'opinions_per_item':
        this_data = raw_data.groupby('item_id').agg({'rating':np.size})['rating']    
        print('Aparecen ' + str(len(this_data)) + ' productos distintos en el dataset.')
        plt.hist(this_data, bins=50, log=True, color='tab:blue')
        print_simple_boxplot(data=this_data, labels=[''], yscale='log', grid='minor')
    
    elif analysis == 'opinions_per_user':
        this_data = raw_data.groupby('user_id').agg({'rating':np.size})['rating']
        print('Aparecen ' + str(len(this_data)) + ' usuarios distintos en el dataset.')
        plt.hist(this_data, bins=50, log=True, color='tab:blue')
        print_simple_boxplot(data=this_data, labels=[''], yscale='log', grid='minor')
            
    elif analysis == 'rating_sd_per_item':
        this_data = raw_data.groupby('item_id').agg({'rating':np.std})
        this_data = this_data.dropna()['rating'] #Esta línea sobra si se toman todas las opiniones, pues no habrá nans
        print_simple_boxplot(data=this_data, labels=[''], yscale='linear', grid='major')
        
    elif analysis == 'rating_distribution':
        this_data = raw_data.groupby('rating').agg({'summary':np.size}).rename(columns={'summary':'total_docs'})['total_docs']
        print_simple_histogram(data_agg=this_data)
        
    elif analysis == 'summary_review_length_comparison_per_rating':
        raw_data = add_lenght_info(raw_data)
        raw_data['ratio_text_summary'] = raw_data.apply(lambda row: row['total_text']/row['summary_length'], axis=1)
        resultados_agregados = raw_data.groupby('rating').agg({'summary_length':np.mean, 'review_length':np.mean, 'ratio_text_summary':np.mean, 'total_text':np.mean})
        print_stacked_histogram(resultados_agregados, 'summary_length', '# palabras en summary', 'review_length', '# palabras en review')
        lista_de_ratios = []
        lista_de_ratios.append(raw_data[raw_data.rating==1]['ratio_text_summary'])
        lista_de_ratios.append(raw_data[raw_data.rating==2]['ratio_text_summary'])
        lista_de_ratios.append(raw_data[raw_data.rating==3]['ratio_text_summary'])
        lista_de_ratios.append(raw_data[raw_data.rating==4]['ratio_text_summary'])
        lista_de_ratios.append(raw_data[raw_data.rating==5]['ratio_text_summary'])
        print_simple_boxplot(data = lista_de_ratios, labels=[1, 2, 3, 4, 5], yscale='log', grid='minor')
    elif analysis == 'text_length_per_rating':
        raw_data = add_lenght_info(raw_data)
        lista_de_longitudes = []
        lista_de_longitudes.append(raw_data[raw_data.rating==1]['total_text'])
        lista_de_longitudes.append(raw_data[raw_data.rating==2]['total_text'])
        lista_de_longitudes.append(raw_data[raw_data.rating==3]['total_text'])
        lista_de_longitudes.append(raw_data[raw_data.rating==4]['total_text'])
        lista_de_longitudes.append(raw_data[raw_data.rating==5]['total_text'])
        print_simple_boxplot(data = lista_de_longitudes, labels=[1, 2, 3, 4, 5], yscale='log', grid='minor')
########################################################################################################################








########################################################################################################################
def get_opinions_full_df(nrows, sampling=False):
    df = pd.read_csv (r'opinions_full_df_1000000.csv')
    if sampling:
        return df.sample(n=nrows, random_state=1)
    return df.loc[:nrows-1]
########################################################################################################################
def build_odf(doc_numbers, ratings, summaries_bigrams, summaries_raw, reviews_bigrams, reviews_raw, corpus_aceptado=None):
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN BUILD_ODF')
    text_list_of_strings = []
    text = []
    text2 = []
    for i in range(0, len(summaries_bigrams)):
        text.append(summaries_bigrams[i]+reviews_bigrams[i])
        if i%10000 == 0:
           calcula_y_muestra_tiempos_2('BUCLE GENERACIÓN VARIABLE TEXT: i='+str(i)+' DE '+str(len(summaries_bigrams)))
    calcula_y_muestra_tiempos_2('VARIABLE TEXT GENERADA')
    
    if corpus_aceptado is not None and len(corpus_aceptado) > 0:
        calcula_y_muestra_tiempos_2('SE HA PROPORCIONADO CORPUS_ACEPTADO')
        calcula_y_muestra_tiempos_2('SE PROCEDE A GENERERAR LA VARIABLE TEXT APLICANDDO CORPUS_ACEPTADO')
        corpus_aceptado = list(corpus_aceptado['bigram'])
        i = 0
        for linea in text:
            linea2 = []
            for item in linea:
                if item in corpus_aceptado:
                    linea2.append(item)
            text2.append(linea2)
            i += 1
            if i%10000 == 0:
                calcula_y_muestra_tiempos_2('BUCLE GENERACIÓN VARIABLE TEXT: i='+str(i)+' DE '+str(len(text)))
        calcula_y_muestra_tiempos_2('VARIABLE TEXT GENERADA')
        for linea2 in text2:
            text_list_of_strings.append(', '.join(map(str, linea2)))
        calcula_y_muestra_tiempos_2('VARIABLE TEXT TRANSFORMADA')
    else:
        for linea in text:
            text_list_of_strings.append(', '.join(map(str, linea)))
        calcula_y_muestra_tiempos_2('VARIABLE TEXT TRANSFORMADA')
    
    summary_tokens_length = [len(x) for x in summaries_bigrams]
    
    tmp = pd.DataFrame(zip(doc_numbers, ratings, text_list_of_strings, summaries_raw, reviews_raw, summary_tokens_length), columns=['doc_number', 'rating', 'text', 'summary_raw', 'review_raw', 'summary_tokens_length'])
    tmp.index = tmp['doc_number']
    
    retorno = tmp[['doc_number', 'rating', 'text', 'summary_raw', 'review_raw', 'summary_tokens_length']]
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN BUILD_ODF')
    
    return retorno
########################################################################################################################
def sent_to_words(sentences):
    retorno = []
    for sentence in sentences:
        retorno.append(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations     
    return retorno
########################################################################################################################
def remove_stopwords(texts):
    '''
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    # las siguientes palabras se han incluido aquí habiendo hecho un bow y revisando palabras frecuentes de poco valor
    stop_words.extend(['one', 'get', 'got', 'getting', 'shoud', 'like', 'also', 'would', 'even', 'could', 'two', 'item', 'thing', 'put', 'however',
                       'something', 'etc', 'unless', 'https', 'www'])
    palabras_deseadas = ['too', 'not', 'no']
    for palabra in palabras_deseadas:
        stop_words.remove(palabra)'''
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
def analize_wordset_not_so_naive(df, wordset_wrapper, show=False):
    
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN ANALIZE_WORDSET_NOT_SO_NAIVE')
    max_distance = 3
    wordset_ands = wordset_wrapper['wordset']['ands']
    wordset_ors = wordset_wrapper['wordset']['ors']
    wordset_name = wordset_wrapper['name']
    
    rows = []
    
    calcula_y_muestra_tiempos_2('ARRANCA EL BUCLE DE OPINIONES')
    for opinion in df.iterrows():
        a_row = {'doc_number':0}
        opinion_text = opinion[1]['text'].split(', ')
        ands_count = 0
        for and_word in wordset_ands:
            if opinion_text.count(and_word) > 0:
                ands_count += 1
                
        if ands_count == len(wordset_ands): # se cumplen los ands
            or_groups_count = 0
            for or_group in wordset_ors:
                for syn0_word in or_group['syn0']:
                    if syn0_word in opinion_text:
                        or_groups_count += 1
                        break
                for syn1_word in or_group['syn1']:
                    if syn1_word in opinion_text:
                        matches = [opinion_text.index(syn1_word)]
                        if len(matches) > 0:
                            for match in matches:
                                start = match-max_distance
                                if match == 0:
                                    start = 0
                                close_words = opinion_text[start:match+max_distance+1]
                                if len(or_group['syn2']) > 0:
                                    for syn2_word in or_group['syn2']:
                                        if close_words.count(syn2_word) > 0:
                                            or_groups_count += 1
                                            break
                                else:
                                    or_groups_count += 1
                                    break
            if or_groups_count >= len(wordset_ors):
                a_row = {'doc_number': opinion[1]['doc_number']}
                rows.append(a_row)
        if opinion[1]['doc_number'] % 50000 == 0:
            calcula_y_muestra_tiempos_2('BUCLE OPINIONES: NUM_DOCUMENTO='+str(opinion[1]['doc_number'])+' DE '+str(len(df)))
    calcula_y_muestra_tiempos_2('FINALIZA EL BUCLE DE OPINIONES')
    rows = pd.DataFrame(rows)
    
    if len(rows) == 0:
        df_filtered = df.iloc[0:0]
        resultados_agregados = df[['rating', 'doc_number']].iloc[0:0]
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
            
        #resultados_agregados['rating_c'] = resultados_agregados.index.values.astype('int64')
        #totales = df.groupby('rating').agg({'doc_number': np.size})
        #totales['rating_c'] = totales.index.values.astype('int64')
        #totales = totales.rename(columns={"doc_number": "total_docs"})
            
        #resultados_agregados = pd.merge(resultados_agregados, totales, on='rating_c')
        #resultados_agregados = resultados_agregados.set_index('rating_c')
        #resultados_agregados['ratio_docs'] = round(resultados_agregados['doc_number']/resultados_agregados['total_docs'], 5)
        
        if show:
            #show_aggregated_results_3(resultados_agregados, wordset_name)
            print_simple_histogram(resultados_agregados['doc_number'])
    
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN ANALIZE_WORDSET_NOT_SO_NAIVE')
    return df_filtered, resultados_agregados
########################################################################################################################
def analize_wordset_not_so_naive_2(df, wordset_wrapper, show=False):
    
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN ANALIZE_WORDSET_NOT_SO_NAIVE')
    max_distance = 3
    wordset_ands = wordset_wrapper['wordset']['ands']
    wordset_ors = wordset_wrapper['wordset']['ors']
    wordset_name = wordset_wrapper['name']
    
    rows = []
    
    calcula_y_muestra_tiempos_2('ARRANCA EL BUCLE DE OPINIONES')
    ##################################################_C_O_R E_########################################
    i = 0
    for opinion in df.iterrows():
        opinion_text = opinion[1]['text'].split(', ')
        #opinion_text = ['this', 'is', 'a', 'random', 'not', 'easy_install', 'totally', 'installation', 'easy', 'testing', 'purposes']
        
        ands_compliance = True
        for and_word in wordset_ands:
            if opinion_text.count(and_word) == 0:
                ands_compliance = False
        
        if ands_compliance:
            ors_compliance = False
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
                    
                    if len(syn0_indices)+len(syn1_indices) == 0:
                        break # Salimos del bucle for or_group in wordset_ors porque este no se satisface
                    else:
                        syn0_array = [0] * len(opinion_text)
        
                        for syn0_indice in syn0_indices:
                            syn0_array[syn0_indice] = 1
                        
                        syn1_array = [0] * len(opinion_text)
        
                        for syn1_indice in syn1_indices:
                            syn1_array[syn1_indice] = 1
                        
                        syn2_words = or_group['syn2']
                        syn2_indices = []
                        syn2_array = [0] * len(opinion_text)
                        for syn2_word in syn2_words:
                            syn2_indices.extend([i for i, j in enumerate(opinion_text) if j == syn2_word])
                        for syn2_indice in syn2_indices:
                            syn2_array[syn2_indice] = 1
                        
                        nots_words = or_group['nots']
                        nots_indices = []
                        nots_array = [0] * len(opinion_text)
                        for nots_word in nots_words:
                            nots_indices.extend([i for i, j in enumerate(opinion_text) if j == nots_word])
                        for nots_indice in nots_indices:
                            nots_array[nots_indice] = 1  
                        
                        matrix = pd.DataFrame(zip(syn0_array, syn1_array, syn2_array, nots_array)).transpose()
                        matrix.columns = opinion_text
                        matrix.index = ['syn0', 'syn1', 'syn2', 'nots']
                        
                        cs_syn0_ok = matrix.loc['syn0'].sum()
                        cs_syn1_ok = 0
                        cs_syn0_ko = 0
                        cs_syn1_ko = 0
                        
                        for indice in syn0_indices:
                            cs_syn0_ko += matrix.loc['nots'][indice-max_distance:indice+max_distance].sum()
                    
                        if cs_syn0_ko == 0:
                            syn1_indices_sobreviven = []
                            for indice in syn1_indices:
                                if matrix.loc['syn2'][indice-max_distance:indice+max_distance].sum() > 0:
                                    syn1_indices_sobreviven.append(indice)
                            
                            syn1_array_sobreviven = [0] * len(opinion_text)
                            for syn1_indice in syn1_indices_sobreviven:
                                syn1_array_sobreviven[syn1_indice] = 1
                            
                            matrix = pd.DataFrame(zip(syn0_array, syn1_array_sobreviven, nots_array)).transpose()
                            matrix.columns = opinion_text
                            matrix.index = ['syn0', 'syn1', 'nots']
                            
                            cs_syn1_ok = matrix.loc['syn1'].sum()
                            if cs_syn1_ok > 0:
                                for indice in syn1_indices_sobreviven:
                                    cs_syn1_ko += matrix.loc['nots'][indice-max_distance:indice+max_distance].sum()
                    
                        if (cs_syn0_ok > 0 or cs_syn1_ok > 0) and cs_syn0_ko + cs_syn1_ko == 0: #El grupo or cumple
                            ors_compliance = True
                            rows.append({'doc_number': opinion[1]['doc_number']})
    ##################################################_C_O_R_E_########################################
        if i % 50000 == 0:
            calcula_y_muestra_tiempos_2('BUCLE OPINIONES: NUM_DOCUMENTO='+str(i)+' DE '+str(len(df)))
        i += 1
    calcula_y_muestra_tiempos_2('FINALIZA EL BUCLE DE OPINIONES')
    rows = pd.DataFrame(rows)
    
    if len(rows) == 0:
        df_filtered = df.iloc[0:0]
        resultados_agregados = df[['rating', 'doc_number']].iloc[0:0]
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
            
        #resultados_agregados['rating_c'] = resultados_agregados.index.values.astype('int64')
        #totales = df.groupby('rating').agg({'doc_number': np.size})
        #totales['rating_c'] = totales.index.values.astype('int64')
        #totales = totales.rename(columns={"doc_number": "total_docs"})
            
        #resultados_agregados = pd.merge(resultados_agregados, totales, on='rating_c')
        #resultados_agregados = resultados_agregados.set_index('rating_c')
        #resultados_agregados['ratio_docs'] = round(resultados_agregados['doc_number']/resultados_agregados['total_docs'], 5)
        
        if show:
            #show_aggregated_results_3(resultados_agregados, wordset_name)
            print_simple_histogram(resultados_agregados['doc_number'])
    
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN ANALIZE_WORDSET_NOT_SO_NAIVE')
    return df_filtered, resultados_agregados
########################################################################################################################
########################################################################################################################
def analize_wordset_not_so_naive_3(df, wordset_wrapper, show=False):
    
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN ANALIZE_WORDSET_NOT_SO_NAIVE')
    
    max_distance = 3
    wordset_ands = wordset_wrapper['wordset']['ands']
    wordset_ors = wordset_wrapper['wordset']['ors']
    wordset_name = wordset_wrapper['name']
    calcula_y_muestra_tiempos_2('WORDSET A ANALIZAR: ' + wordset_name)
    
    rows = []
    
    elementos_or = get_wsw_structure(wordset_wrapper)[1]
    
    calcula_y_muestra_tiempos_2('ARRANCA EL BUCLE DE OPINIONES')
    ##################################################_C_O_R_E_########################################
    i = 0
    for opinion in df.iterrows():
        ors_compliance = False
        and_compliance = False
        parte = 'summary'
        
        for pase in range(0,2):
            if parte == 'summary':
                opinion_text = opinion[1]['text'].split(', ')[0:opinion[1]['summary_tokens_length']]
                parte = 'review'
            elif not ors_compliance:
                opinion_text = opinion[1]['text'].split(', ')[opinion[1]['summary_tokens_length']:]
    
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
    ##################################################_C_O_R_E_########################################
        if i % 50000 == 0:
            calcula_y_muestra_tiempos_2('BUCLE OPINIONES: NUM_DOCUMENTO='+str(i)+' DE '+str(len(df)))
        i += 1
    calcula_y_muestra_tiempos_2('FINALIZA EL BUCLE DE OPINIONES')
    rows = pd.DataFrame(rows)
    
    if len(rows) == 0:
        df_filtered = df.iloc[0:0]
        resultados_agregados = df[['rating', 'doc_number']].iloc[0:0]
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
            
        #resultados_agregados['rating_c'] = resultados_agregados.index.values.astype('int64')
        #totales = df.groupby('rating').agg({'doc_number': np.size})
        #totales['rating_c'] = totales.index.values.astype('int64')
        #totales = totales.rename(columns={"doc_number": "total_docs"})
            
        #resultados_agregados = pd.merge(resultados_agregados, totales, on='rating_c')
        #resultados_agregados = resultados_agregados.set_index('rating_c')
        #resultados_agregados['ratio_docs'] = round(resultados_agregados['doc_number']/resultados_agregados['total_docs'], 5)
        
        if show:
            #show_aggregated_results_3(resultados_agregados, wordset_name)
            print_simple_histogram(resultados_agregados['doc_number'], title = "Analize para el wordset '" + wordset_name + "'")
    
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN ANALIZE_WORDSET_NOT_SO_NAIVE')
    return df_filtered, resultados_agregados
########################################################################################################################
def check_existencia (conjunto, tokens_a_buscar):
    for token in tokens_a_buscar:
        if token in conjunto:
            return True
########################################################################################################################
def analize_wordset_not_so_naive_4(df, wordset_wrapper, show=False):
    
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN ANALIZE_WORDSET_NOT_SO_NAIVE')
    
    max_distance = 3
    wordset_ands = wordset_wrapper['wordset']['ands']
    wordset_ors = wordset_wrapper['wordset']['ors']
    wordset_name = wordset_wrapper['name']
    calcula_y_muestra_tiempos_2('WORDSET A ANALIZAR: ' + wordset_name)
    lista_or_words = get_ws_or_words(wordset_wrapper)
    rows = []
    
    elementos_or = get_wsw_structure(wordset_wrapper)[1]
    
    calcula_y_muestra_tiempos_2('ARRANCA EL BUCLE DE OPINIONES')
    ##################################################_C_O_R_E_########################################
    i = 0
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
    ##################################################_C_O_R_E_########################################
        if i % 50000 == 0:
            calcula_y_muestra_tiempos_2('BUCLE OPINIONES: NUM_DOCUMENTO='+str(i)+' DE '+str(len(df)))
        i += 1
    calcula_y_muestra_tiempos_2('FINALIZA EL BUCLE DE OPINIONES')
    rows = pd.DataFrame(rows)
    
    if len(rows) == 0:
        df_filtered = df.iloc[0:0]
        resultados_agregados = df[['rating', 'doc_number']].iloc[0:0]
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
            
        #resultados_agregados['rating_c'] = resultados_agregados.index.values.astype('int64')
        #totales = df.groupby('rating').agg({'doc_number': np.size})
        #totales['rating_c'] = totales.index.values.astype('int64')
        #totales = totales.rename(columns={"doc_number": "total_docs"})
            
        #resultados_agregados = pd.merge(resultados_agregados, totales, on='rating_c')
        #resultados_agregados = resultados_agregados.set_index('rating_c')
        #resultados_agregados['ratio_docs'] = round(resultados_agregados['doc_number']/resultados_agregados['total_docs'], 5)
        
        if show:
            #show_aggregated_results_3(resultados_agregados, wordset_name)
            print_simple_histogram(resultados_agregados['doc_number'], title = "Analize para el wordset '" + wordset_name + "'")
    
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN ANALIZE_WORDSET_NOT_SO_NAIVE')
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
#def show_aggregated_results_2(resultados_agregados, word):
#    import matplotlib.pyplot as plt
#    x_values = resultados_agregados.index.values.astype('int64')
#    fig, ax1 = plt.subplots()
#    plt.title("Ratio documentos en que aparece el elemento '" + word + "'")
#    ax1.set_xlabel('Rating')
#    ax1.bar(x_values-0.2, resultados_agregados['ratio_docs'], color='orange', width=0.4)
#    ax1.set_ylabel('Ratio documentos', color='orange')
#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#    ax2.bar(x_values+0.2, resultados_agregados['doc_number'], color='green', width=0.4)
#    ax2.set_ylabel('# Opiniones', color='green')
#    plt.show()
########################################################################################################################
#def show_aggregated_results_3(resultados_agregados, wsw_name):
#    x_values = resultados_agregados.index.values.astype('int64')
#    plt.ylabel('# Referencias')
#    plt.xlabel('Rating')
#    plt.title("# Menciones del elemento '" + wsw_name + "'")
#    plt.bar(x_values, resultados_agregados['doc_number'], color='blue')
#    plt.show()
########################################################################################################################
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    import spacy
    nlp = spacy.load('en', disable=['parser', 'ner'])
    #"""https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
########################################################################################################################
def execute_preprocessing_pipeline(opinions_raw, eliminar_poco_frecuentes=True, lemmatize=False):

    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN EXECUTE_PREPROCESSING_PIPELINE')
    categorias_gramaticales_admitidas = ['NOUN', 'ADJ', 'VERB', 'ADV']
    summaries_raw = opinions_raw.summary.values.tolist()
    summaries_words = sent_to_words(summaries_raw)
    
    
    #opinions_raw = data_raw_1
    num_docs = opinions_raw.index.values
    
    
    
    i = 0
    for i in range(0, len(summaries_words)):
        #summaries_words[i] = ['not' if x=='didn' or x =='didnt' or x =='doesn' or x=='doesnt' or x=='dont' or x=='don' else x for x in summaries_words[i]]
        summaries_words[i] = mapear_palabras_especiales(summaries_words[i])
    i += 1
    summaries_words_nostop = remove_stopwords(summaries_words)
    summaries_bigrams = make_bigrams(summaries_words_nostop, summaries_words)

    calcula_y_muestra_tiempos_2('SUMMARIES PROCESADOS')
    reviews_raw = opinions_raw.review.values.tolist()
    reviews_words = sent_to_words(reviews_raw)
    i = 0
    for i in range(0, len(reviews_words)):
        #reviews_words[i] = ['not' if x=='didn' or x =='didnt' or x =='doesn' or x=='doesnt' or x=='dont' or x=='don' else x for x in reviews_words[i]]
        reviews_words[i] = mapear_palabras_especiales(reviews_words[i])
    i += 1
    reviews_words_nostop = remove_stopwords(reviews_words)
    reviews_bigrams = make_bigrams(reviews_words_nostop, reviews_words)
    calcula_y_muestra_tiempos_2('REVIEWS PROCESADAS')
    
    if lemmatize:
        reviews_lemmatized = lemmatization(reviews_bigrams, allowed_postags=categorias_gramaticales_admitidas)
        summaries_lemmatized = lemmatization(summaries_bigrams, allowed_postags=categorias_gramaticales_admitidas)
        odf_false = build_odf(num_docs, opinions_raw.rating, summaries_lemmatized, summaries_raw, reviews_lemmatized, reviews_raw, None)
    else: 
        odf_false = build_odf(num_docs, opinions_raw.rating, summaries_bigrams, summaries_raw, reviews_bigrams, reviews_raw, None)
        
    calcula_y_muestra_tiempos_2('ODF_FALSE CONSTRUIDO')
    url_false = r'..\data\odfs\odf_false_'+str(len(odf_false))+'.csv'
    odf_false.to_csv (url_false, index = False, header=True)
    calcula_y_muestra_tiempos_2('ODF_FALSE SALVADO EN DATA/ODFS')
        
    if eliminar_poco_frecuentes:
        odf_true = build_odf(num_docs, opinions_raw.rating, summaries_bigrams, summaries_raw, reviews_bigrams, reviews_raw, generar_corpus_aceptado(odf_false))
        calcula_y_muestra_tiempos_2('ODF CONSTRUIDO DE NUEVO, ELIMINANDO PALABRAS POCO FRECUENTES')
        url = r'..\data\odfs\odf_'+str(len(odf_true))+'.csv'
        
        odf_true.to_csv (url, index = False, header=True)
        calcula_y_muestra_tiempos_2('ODF_TRUE SALVADO EN DATA/ODFS')
        calcula_y_muestra_tiempos_2('FIN FUNCIÓN EXECUTE_PREPROCESSING_PIPELINE')
        
        return odf_false, odf_true

    calcula_y_muestra_tiempos_2('FIN FUNCIÓN EXECUTE_PREPROCESSING_PIPELINE')
    return odf_false
########################################################################################################################
def generate_bow(df, classify_pos_b=True, show=False):
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN GENERATE_BOW')
    bigrams = []
    
    i = 0
    for opinion in df.iterrows():
        text_list = opinion[1]['text']
        if text_list.count(', ') > 0:
            bigrams.extend(text_list.split(', '))
        i += 1
        if i%50000 == 0:
            calcula_y_muestra_tiempos_2('BUCLE GENERANDO BOW: i='+str(i)+' de '+str(len(df)))
    calcula_y_muestra_tiempos_2('LISTA DE BIGRAMS FINALIZADA')
    aux = pd.Series(bigrams).value_counts()
    
    bag = pd.DataFrame(zip(aux.index, aux.values), columns=['bigram', 'num_occurrences']).sort_values('num_occurrences', ascending=False)
    if classify_pos_b:
        bag['pos'] = bag.apply(lambda row: classify_pos(row['bigram']), axis=1)
    if show:
        print_wordcloud_from_frequencies(bag[['bigram', 'num_occurrences']], 18)
        
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN GENERATE_BOW')
    return bag
########################################################################################################################
def print_wordcloud_from_frequencies(bag, max_words=999):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
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
def generar_corpus_aceptado(odf):
    bow=generate_bow(odf, False)
    bow['num_occurrences'][0]
    #bow['ratio'] = bow['num_occurrences']/bow['num_occurrences'][0]   
    #corpus_aceptado = bow[bow['ratio'] >= 0.001]
    corpus_aceptado = bow[bow['num_occurrences'] > 2]
    return corpus_aceptado
########################################################################################################################
def extract_bigrams_from_bow(bow):
    #bow = generate_bow(df=odf, show=False)
    bigrams = []
    number = []
    for item in bow.iterrows():
        if '_' in item[1]['bigram']:
            bigrams.append(item[1]['bigram'])
            number.append(item[1]['num_occurrences'])
    retorno = pd.DataFrame(list(zip(bigrams, number)), columns =['bigram', 'num_occurrences'])
    
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
def calcula_y_muestra_tiempos(texto):
    timestamps.append(datetime.now())
    tot = len(timestamps)
    print('::::', texto, '::::', strfdelta(timestamps[tot-1]-timestamps[tot-2]))
########################################################################################################################
def get_ws_words(wsw):
    wordset_ands = wsw['wordset']['ands']
    wordset_ors = wsw['wordset']['ors']
    words_list = []
    for and_word in wordset_ands:
        words_list.append(and_word)
    for or_group in wordset_ors:
        for syn0_word in or_group['syn0']:
            words_list.append(syn0_word)
        for syn1_word in or_group['syn1']:
            words_list.append(syn1_word)
        for syn2_word in or_group['syn2']:
            words_list.append(syn2_word)
    return words_list
########################################################################################################################
def generate_bow_post_analize(odf_filtered, wsw, show=False):
    bow = generate_bow(odf_filtered, show=False)
    ws_words = get_ws_words(wsw)
    rows_to_delete = []
    i = 0
    for i in range(0, len(bow)):
        if bow['bigram'][i] in ws_words:
            rows_to_delete.append(i)
    
    retorno = bow.drop(bow.index[rows_to_delete])
    
    if show:
        print_wordcloud_from_frequencies(retorno[['bigram', 'num_occurrences']], 18)
    return retorno
########################################################################################################################
def remove_stopwords_from_bow(bow, stopwords_modo):
    rows_to_delete = []
    i = 0
    for i in range(0, len(bow)):
        if bow['bigram'][i] in get_stopwords(modo=stopwords_modo):
            rows_to_delete.append(i)
    
    retorno = bow.drop(rows_to_delete)
    return retorno
########################################################################################################################
def generate_bow_not_so_naive(odf_filtered, wsw, stopwords, show=False):
    bow = generate_bow(odf_filtered, show=False)
    ws_words = get_ws_words(wsw)
    rows_to_delete = []
    i = 0
    for i in range(0, len(bow)):
        if bow['bigram'][i] in ws_words+stopwords:
            rows_to_delete.append(i)
    
    retorno = bow.drop(rows_to_delete)
    
    if show:
        print_wordcloud_from_frequencies(retorno[['bigram', 'num_occurrences']], 18)
    return retorno
########################################################################################################################
def generate_bow_post_analize_main(odf_filtered, wsw, show=False):
    print('Análisis independiente de la puntuación a continuación')
    bowF = generate_bow_post_analize(odf_filtered, wsw, show)
    #bow1, bow2, bow3, bow4, bow5 = pd.DataFrame()
    print('Análisis para 1 estrella a continuación')
    if len(odf_filtered[odf_filtered['rating']==1]) > 0:
        bow1 = generate_bow_post_analize(odf_filtered[odf_filtered['rating']==1], wsw, show)
    else:
        bow1 = pd.DataFrame()
    print('Análisis para 2 estrellas a continuación')
    if len(odf_filtered[odf_filtered['rating']==2]) > 0:
        bow2 = generate_bow_post_analize(odf_filtered[odf_filtered['rating']==2], wsw, show)
    else:
        bow2 = pd.DataFrame()
    print('Análisis para 3 estrellas a continuación')
    if len(odf_filtered[odf_filtered['rating']==3]) > 0:
        bow3 = generate_bow_post_analize(odf_filtered[odf_filtered['rating']==3], wsw, show)
    else:
        bow3 = pd.DataFrame()
    print('Análisis para 4 estrellas a continuación')
    if len(odf_filtered[odf_filtered['rating']==4]) > 0:
        bow4 = generate_bow_post_analize(odf_filtered[odf_filtered['rating']==4], wsw, show)
    else:
        bow4 = pd.DataFrame()
    print('Análisis para 5 estrellas a continuación')
    if len(odf_filtered[odf_filtered['rating']==5]) > 0:
        bow5 = generate_bow_post_analize(odf_filtered[odf_filtered['rating']==5], wsw, show)
    else:
        bow5 = pd.DataFrame()
    return {'F':bowF, '1':bow1, '2':bow2, '3':bow3, '4':bow4, '5':bow5}
########################################################################################################################
def generate_bow_not_so_naive_main(odf_filtered, wsw, stopwords, show=False):
    print('Análisis independiente de la puntuación a continuación')
    bowF = generate_bow_not_so_naive(odf_filtered, wsw, stopwords, show)
    #bow1, bow2, bow3, bow4, bow5 = pd.DataFrame()
    print('Análisis para 1 estrella a continuación')
    if len(odf_filtered[odf_filtered['rating']==1]) > 0:
        bow1 = generate_bow_not_so_naive(odf_filtered[odf_filtered['rating']==1], wsw, stopwords, show)
    else:
        bow1 = pd.DataFrame()
    print('Análisis para 2 estrellas a continuación')
    if len(odf_filtered[odf_filtered['rating']==2]) > 0:
        bow2 = generate_bow_not_so_naive(odf_filtered[odf_filtered['rating']==2], wsw, stopwords, show)
    else:
        bow2 = pd.DataFrame()
    print('Análisis para 3 estrellas a continuación')
    if len(odf_filtered[odf_filtered['rating']==3]) > 0:
        bow3 = generate_bow_not_so_naive(odf_filtered[odf_filtered['rating']==3], wsw, stopwords, show)
    else:
        bow3 = pd.DataFrame()
    print('Análisis para 4 estrellas a continuación')
    if len(odf_filtered[odf_filtered['rating']==4]) > 0:
        bow4 = generate_bow_not_so_naive(odf_filtered[odf_filtered['rating']==4], wsw, stopwords, show)
    else:
        bow4 = pd.DataFrame()
    print('Análisis para 5 estrellas a continuación')
    if len(odf_filtered[odf_filtered['rating']==5]) > 0:
        bow5 = generate_bow_not_so_naive(odf_filtered[odf_filtered['rating']==5], wsw, stopwords, show)
    else:
        bow5 = pd.DataFrame()
    return {'F':bowF, '1':bow1, '2':bow2, '3':bow3, '4':bow4, '5':bow5}
########################################################################################################################
def analize_wordset_occurrences(lista_resultados_analize):
    matriz_documento_wordset = pd.DataFrame(index=odf.index)
    for resultado_busqueda in lista_resultados_busquedas:
        nombre = resultado_busqueda['name']
        matriz_documento_wordset[nombre] = 0
        for doc_number in resultado_busqueda['resultados']['doc_number']:
            matriz_documento_wordset[nombre][doc_number] = 1
    matriz_documento_wordset['total_wordsets'] = matriz_documento_wordset.sum(axis=1)
    ratio_ocupacion_total = matriz_documento_wordset.sum().sum()/(matriz_documento_wordset.size-len(matriz_documento_wordset))
    ratio_ocupacion = np.count_nonzero(matriz_documento_wordset['total_wordsets'])/len(matriz_documento_wordset)
    matriz_documento_wordset_agg = matriz_documento_wordset.groupby('total_wordsets').count()[matriz_documento_wordset.groupby('total_wordsets').count().columns[0]].to_frame()
    matriz_documento_wordset_agg.columns.values[0] = 'total_documents'
    print_simple_histogram(matriz_documento_wordset_agg['total_documents'], title = '# de opiniones por hits')
    print ('La ocupación parcial es del', "{:.0%}".format(ratio_ocupacion))
    print ('La ocupación total es del', "{:.0%}".format(ratio_ocupacion_total))
    
    return matriz_documento_wordset, matriz_documento_wordset_agg
########################################################################################################################
def classify_pos(token):
    if token == 'works':
        pos = 'VBZ'
    else:
        pos = nltk.pos_tag([token])[0][1]

    correspondencias =  {
                        'NN': 'sustantivo',
                        'NNS': 'sustantivo',
                        'JJ': 'adjetivo_o_numeral',
                        'JJS': 'adjetivo_superlativo',
                        'JJR': 'adjetivo_comparativo',
                        'VB': 'verbo',
                        'VBN': 'verbo',
                        'VBG': 'verbo',
                        'VBZ': 'verbo',
                        'VBD': 'verbo',
                        'RB': 'adverbio',
                        'RBR': 'adverbio_comparativo',
                        'IN': 'preposicion_o_conjuncion',
                        'DT': 'determinante',
                        'MD': 'auxiliar_modal',
                        'CD': 'numeral',                    
                        'PRP': 'pronombre_personal',
                        'CC': 'conjuncion',
                        'WDT': 'determinante_wh',
                        'WP$': 'pronombre_posesivo'
                        }
    pos = correspondencias[pos]
    return pos
########################################################################################################################
def get_stopwords(modo='pre'):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['one', 'get', 'got', 'getting', 'shoud', 'like', 'also', 'would', 'even', 'could', 'two', 'item', 'thing', 'put', 'however',
                       'something', 'etc', 'unless', 'https', 'www', 'for'])
    if modo=='pre':
        palabras_deseadas = ['too', 'not', 'no', 'than', 'out']
        for palabra in palabras_deseadas:
            stop_words.remove(palabra)
    return stop_words
########################################################################################################################
'''def get_close_words(opinion_text, match, max_distance):
    if match == 0:
        start = 0
    else:
        start = match-max_distance
    return opinion_text[start:match+max_distance+1]'''
########################################################################################################################
def mapear_palabras_especiales(array_input):
    retorno = ['not'    if x == 'didn' 
                        or x == 'didnt' 
                        or x == 'doesn' 
                        or x == 'doesnt' 
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
def get_close_words(df, max_distance, word, n_words):
    from collections import Counter
    palabras_antes = []
    palabras_despues  = []

    for opinion in df.iterrows():
        texto = opinion[1]['text'].split(', ')
        if word in set(texto):
            word_indices = []
            word_indices.extend([i for i, j in enumerate(texto) if j == word])
            for indice in word_indices:
                palabras_despues.extend(texto[indice+1:indice+max_distance+1])
                palabras_antes.extend(texto[indice-max_distance:indice])

    palabras_antes_final = []
    palabras_antes_agg = Counter(palabras_antes)
    palabras_antes_agg = remove_stopwords_from_bow(pd.DataFrame(zip(palabras_antes_agg.keys(), palabras_antes_agg.values()), columns=['bigram', 'num_occurrences']), stopwords_modo='post').sort_values(by=['num_occurrences'], ascending=False)
    for palabra_antes in palabras_antes_agg['bigram'].iloc[:n_words]:
        palabras_antes_final.append(palabra_antes)

    palabras_despues_final = []
    palabras_despues_agg = Counter(palabras_despues)
    palabras_despues_agg = remove_stopwords_from_bow(pd.DataFrame(zip(palabras_despues_agg.keys(), palabras_despues_agg.values()), columns=['bigram', 'num_occurrences']), stopwords_modo='post').sort_values(by=['num_occurrences'], ascending=False)
    for palabra_despues in palabras_despues_agg['bigram'].iloc[:n_words]:
        palabras_despues_final.append(palabra_despues)

    return palabras_antes_final, palabras_despues_final
########################################################################################################################
def visualize_wordsets_network(matriz_doc_ws_expanded, ratings='F'):
    # https://www.kaggle.com/jncharon/python-network-graph
    if ratings != 'F':
        matriz_doc_ws_expanded = matriz_doc_ws_expanded[matriz_doc_ws_expanded['rating'].isin(ratings)]
    
    n_puntos_asumibles = 20000
    n_opiniones = len(matriz_doc_ws_expanded) 
    new_len = n_puntos_asumibles if n_opiniones > n_puntos_asumibles else n_opiniones
    matriz_doc_ws_expanded = matriz_doc_ws_expanded.sample(n=new_len)
    
    array_num_doc = []
    array_wordsets = []
    columnas = matriz_doc_ws_expanded.columns.values.tolist()
    wordsets_names = columnas[:len(columnas)-2]
    num_wordsets = len(wordsets_names)
    for opinion in matriz_doc_ws_expanded.iterrows():
        for wordset in wordsets_names:
            if opinion[1][wordset] == 1:
                array_num_doc.append(opinion[0])
                array_wordsets.append(wordset)
    edgelist = pd.DataFrame(zip(array_num_doc, array_wordsets), columns=['from', 'to'])
    edgelist = pd.merge(edgelist, matriz_doc_ws_expanded['rating'], left_on='from', right_index=True, how='inner')
    nodelist = wordsets_names.copy()
    nodelist.extend(edgelist['from'].unique())
    nodelist = pd.DataFrame(nodelist, columns=['node_name'])
    
    sizes = matriz_doc_ws[wordsets_names].sum().tolist()
    ct = 2000/max(sizes)
    sizes = [size * ct for size in sizes]
    sizes.extend((len(nodelist)-num_wordsets)*[30])
    nodelist['size'] = sizes

    grupos = np.zeros(num_wordsets).tolist()
    grupos.extend(matriz_doc_ws_expanded[matriz_doc_ws['total_wordsets'] > 0]['rating'])
    nodelist['grupo'] = grupos
    G = nx.Graph()
    
    color_map = {0:'#E6E6E6', 1:'#cc3232', 2:'#db7b2b', 3:'#e7b416', 4:'#99c140', 5:'#2dc937'}
    
    # NODES
    for index, row in nodelist.iterrows():
        G.add_node(row['node_name'], group=row['grupo'], color=color_map[row['grupo']], size=row['size'])
    
    # EDGES
    for index, row in edgelist.iterrows():
        G.add_edge(row['from'], row['to'], color=color_map[row['rating']])
    
    labels = {}
    for node in nodelist['node_name'][:num_wordsets]:
        labels[node] = node
                
    plt.figure(figsize=(25, 25))
    pos = nx.spring_layout(G, k=0.01, iterations=50)
    edges, edge_colors = zip(*nx.get_edge_attributes(G, 'color').items())
    nodes, node_colors = zip(*nx.get_node_attributes(G, 'color').items())
    nodes, node_sizes = zip(*nx.get_node_attributes(G, 'size').items())
    
    nx.draw(G, pos=pos, nodelist=nodes, node_size=node_sizes, node_color=node_colors, edgelist=edges, edge_color=edge_colors, width=0.4)
    nx.draw_networkx_labels(G, pos, labels, font_size=20)
    plt.show()
########################################################################################################################
def analize_advanced(odf, words, bow):
    wsw =   {
                'name': 'resultados para el token ' + "'" + word + "'"
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        {
                        'syn0': busca_tokens(bow, words),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ]
                }
            }
    an_fil, an_agg = analize_wordset_not_so_naive_3(odf, wsw, show=True)
    return an_fil, an_agg
########################################################################################################################
'''def busca_tokens(bow, words):
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN BUSCA_TOKENS')
    tokens_a_retornar = []
    for word in words:
        for elemento in bow.iterrows():
            if elemento[1]['bigram'].split('_').count(word):
                tokens_a_retornar.append(elemento[1]['bigram'])
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN GENERATE_BOW')
    return tokens_a_retornar'''
########################################################################################################################
def busca_tokens(bow, words):
    calcula_y_muestra_tiempos_2('INICIO FUNCIÓN BUSCA_TOKENS')
    tokens_a_retornar = []
    for word in words:
        for elemento in set(bow['bigram']): # como set el rendimiento mejora espectacularmente
            if elemento.split('_').count(word):
                tokens_a_retornar.append(elemento)
    calcula_y_muestra_tiempos_2('FIN FUNCIÓN GENERATE_BOW')
    return tokens_a_retornar



busca_tokens(tokens, ['house'])


get_close_words(odf, 3, 'user_friendly', 5)





########################################################################################################################


timestamps = []
data_raw_0 = electronics_5_to_raw_data_0(1000000)

perform_eda(data_raw_0, 'rating_distribution')
perform_eda(data_raw_0, 'opinions_per_year')
perform_eda(data_raw_0, 'opinions_per_item')
perform_eda(data_raw_0, 'opinions_per_user')
perform_eda(data_raw_0, 'rating_sd_per_item')
perform_eda(data_raw_0, 'summary_review_length_comparison_per_rating')
perform_eda(data_raw_0, 'text_length_per_rating')


data_raw_1 = remove_rating_bias_from_raw_data(data_raw_0)

perform_eda(data_raw_1, 'rating_distribution')
perform_eda(data_raw_1, 'opinions_per_year')
perform_eda(data_raw_1, 'opinions_per_item')
perform_eda(data_raw_1, 'opinions_per_user')
perform_eda(data_raw_1, 'rating_sd_per_item')
perform_eda(data_raw_1, 'summary_review_length_comparison_per_rating')
perform_eda(data_raw_1, 'text_length_per_rating')


odf_false, odf = execute_preprocessing_pipeline(data_raw_1)
odf = execute_preprocessing_pipeline(data_raw_1, False)
odf = load_latest_odf(nrows=214475, is_false=True)
timestamps = []
#odf = load_latest_odf(nrows=1532805, is_false=False)
#odf_false = load_latest_odf(nrows=1532805, is_false=True)


























tokens = generate_bow(odf, False, show=False)
bigrams = extract_bigrams_from_bow(tokens)

wsw1 =   {
            'name':'buen funcionamiento del producto'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    # MUY POSITIVO
                    {
                    'syn0': ['works_great', 'works_flawlessly', 'works_perfectly', 'worked_perfectly', 'works_well'],
                    'syn1': [],
                    'syn2': [],
                    'nots': ['not']
                    }
                    # POSITIVO
                    ,{
                    'syn0': ['works_fine', 'good_job', 'working_properly', 'works_ok', 'work_properly', 'serves_purpose'],
                    'syn1': [],
                    'syn2': [],
                    'nots': ['not']
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['works'],
                    'syn2': ['expected'],
                    'nots': ['not']
                    }
                    # BAJO CONSUMO
                    ,{
                    'syn0': [],
                    'syn1': ['power_consumption'],
                    'syn2': ['low'],
                    'nots': []
                    }
                    ]
            }
        }
an1_fil, an1_agg = analize_wordset_not_so_naive_4(odf, wsw1, True)

wsw2 =   {
            'name':'mal funcionamiento del producto'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    # GENÉRICO, NO FUNCIONA DEBIDAMENTE
                    {
                    'syn0': ['not_work'],
                    'syn1': ['working_properly', 'work_properly'],
                    'syn2': ['not'],
                    'nots': []
                    }
                    ,{
                    'syn0': ['faulty', 'defective', 'fluke'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    # PROBLEMAS DE ESTABILIDAD
                    ,{
                    'syn0': ['unstable', 'stability_issues'],
                    'syn1': ['stable'],
                    'syn2': ['not'],
                    'nots': []
                    },
                    {
                    'syn0': [],
                    'syn1': ['suddenly_stopped', 'stopped'],
                    'syn2': ['suddenly', 'working'],
                    'nots': []
                    },
                    {
                    'syn0': ['unreliable', 'works_intermittently'],
                    'syn1': ['reliable'],
                    'syn2': ['not'],
                    'nots': []
                    }
                    # DEJÓ DE FUNCIONAR O COMENZÓ A FUNCIONAR MAL
                    ,{
                    'syn0': [],
                    'syn1': ['quit_working', 'stopped_working', 'stopped_working', 'stop_working', 'quits_working', 'quitted_working'],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['within_week', 'within_weeks', 'week', 'weeks'],
                    'syn2': ['stopped_working', 'started', 'broke', 'quits_working', 'quitted_working'],
                    'nots': []
                    }
                    ,{
                    'syn0': ['obsolescence', 'short_life', 'never_worked'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    # ALTO CONSUMO
                    ,{
                    'syn0': ['drains_battery', 'eat_batteries', 'eats_batteries', 'eating_batteries'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['power_consumption'],
                    'syn2': ['high'],
                    'nots': []
                    },
                    ]
            }
        }
an2_fil, an2_agg = analize_wordset_not_so_naive_4(odf, wsw2, True)

wsw3 =   {
            'name':'característica del producto'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    # GENÉRICO APARIENCIA
                    {
                    'syn0': busca_tokens(tokens, ['looks']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    # MALA CALIDAD DE PIEZAS MECÁNICAS Y COMPONENTES
                    ,{
                    'syn0': busca_tokens(tokens, ['defective']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': ['broke_apart', 'fell_apart', 'fall_apart', 'falling_apart',  'cheaply_made', 'cheap_plastic', 'poor_quality', 'bit_flimsy', 'defective_product', 'poorly_constructed'],
                    'syn1': ['melted'],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': ['feels_cheap', 'poorly_made'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['build_quality'],
                    'syn2': ['bad'],
                    'nots': ['not']
                    }
                    ,{
                    'syn0': ['broke_within', 'easily_broken', 'breaks_easily', 'breaks_easy', 'breaks_every', 'breaks_quickly', 'broke_quickly', 'broke_shortly'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['sound_quality', 'audio_quality'],
                    'syn2': ['poor', 'bad'],
                    'nots': []
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['picture_quality', 'image_quality'],
                    'syn2': ['poor', 'bad'],
                    'nots': []
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['battery_life'],
                    'syn2': ['short'],
                    'nots': []
                    }
                    # DISEÑO PRESENTA FALLOS
                    ,{
                    'syn0': ['design_flaw', 'design_defect', 'design_weakness', 'flawed_design', 'poor_design'],
                    'syn1': ['serious_design'],
                    'syn2': ['flaw', 'flaws'],
                    'nots': []
                    }
                    # TAMAÑO INADECUADO
                    ,{
                    'syn0': [],
                    'syn1': ['fit', 'fits'],
                    'syn2': ['not', 'no'],
                    'nots': []
                    },
                    {
                    'syn0': [],
                    'syn1': ['big', 'small', 'long', 'short'],
                    'syn2': ['too', 'not_enough', 'way_too'],
                    'nots': []
                    },
                    {
                    'syn0': [],
                    'syn1': ['size', 'length'],
                    'syn2': ['wrong'],
                    'nots': ['no']
                    },
                    {
                    'syn0': ['too_short', 'too_long', 'not_fit', 'too_small', 'too_big', 'too_large'],
                    'syn1': ['long_enough'],
                    'syn2': ['not'],
                    'nots': []
                    }
                    # BUENA CALIDAD DE PIEZAS MECÁNICAS Y COMPONENTES
                    ,{
                    'syn0': ['excellent_build', 'sturdy'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': ['well_made', 'quite_good', 'well_designed', 'excellent_product', 'high_quality'],
                    'syn1': ['build_quality', 'image_quality'],
                    'syn2': ['good', 'nice', 'high'],
                    'nots': []
                    },
                    {
                    'syn0': ['excellent_product', 'excellent_quality', 'excellent_results', 'excellent_sound'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': ['sounds_great'],
                    'syn1': ['sound_quality', 'audio_quality'],
                    'syn2': ['good', 'great', 'excellent'],
                    'nots': ['not']
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['picture_quality', 'image_quality'],
                    'syn2': ['good', 'great', 'excellent'],
                    'nots': ['not']
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['battery_life'],
                    'syn2': ['good', 'great', 'long', 'excellent'],
                    'nots': ['not']
                    }
                    # TAMAÑO ADECUADO
                    ,{
                    'syn0': ['fit_perfectly'],
                    'syn1': ['fit', 'fits'],
                    'syn2': ['good', 'perfectly', 'right'],
                    'nots': ['expected', 'not']
                    }
                    ,{
                    'syn0': ['perfect_size'],
                    'syn1': ['size', 'length'],
                    'syn2': ['perfect'],
                    'nots': []
                    }
                    ]
            }
        }               
an3_fil, an3_agg = analize_wordset_not_so_naive_4(odf, wsw3, True)

wsw4 =  {
            'name':'facilidad de uso o instalación'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    # PUESTA EN FUNCIONAMIENTO SENCILLA O RÁPIDA
                    
                    #{
                    #'syn0': ['easy_install', 'easy_installation', 'easy_setup', 'installed_easily', 'quick_installation', 'plug_play'],
                    #'syn1': ['plug'],
                    #'syn2': ['play'],
                    #'nots': []
                    ##'nots': ['not', 'no']
                    #}
                    #,{
                    #'syn0': [],
                    #'syn1': ['installation', 'install', 'setup', 'installed'],
                    #'syn2': ['easy', 'fast', 'quick', 'easily', 'smooth', 'breeze'],
                    #'nots': []
                    ##'nots': ['not', 'no']
                    #}
                    #,{
                    #'syn0': [],
                    #'syn1': ['installed'],
                    #'syn2': ['easily', 'quickly'],
                    #'nots': []
                    ##'nots': ['not', 'no']
                    #}
                    {
                    'syn0': busca_tokens(tokens, ['install', 'installation', 'installed', 'setup']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': ['plug_play'],
                    'syn1': ['plug'],
                    'syn2': ['play'],
                    'nots': []
                    }
                    # USO DEL MANUAL DE INSTRUCCIONES
                    ,{
                    #'syn0': ['followed_instructions', 'written_instructions', 'written_documentation', 'instruction_book', 'instruction_booklet', 'instruction_manual', 'instruction_manuals', 'instruction_sheet', 'instructions_included', 'instructions_say'],
                    'syn0': busca_tokens(tokens, ['instructions']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': busca_tokens(tokens, ['instruction']),
                    #'syn1': ['instructions'],
                    #'syn2': ['understand', 'unintelligible'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    # PRODUCTO DE DIFÍCIL USO, INCÓMODOS O CON ELEMENTOS DESAGRADABLES
                    ,{
                    'syn0': [],
                    'syn1': ['user_friendly'],
                    'syn2': ['not'],
                    'nots': []
                    }
                    ,{
                    'syn0': ['not_recognize'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': ['tricky', 'figure_out', 'figured_out', 'figuring_out'],
                    'syn1': ['trying'],
                    'syn2': ['work'],
                    'nots': []
                    }
                    ,{
                    'syn0': busca_tokens(tokens, ['annoying']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': busca_tokens(tokens, ['uncomfortable']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an4_fil, an4_agg = analize_wordset_not_so_naive_4(odf, wsw4, True)
busca_tokens(tokens, ['frustrating'])
wsw5 =  {
            'name':'expectativas vs realidad'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': [],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an5_fil, an5_agg = analize_wordset_not_so_naive_4(odf, wsw5, True)
      
wsw6 =  {
            'name':'referencia a opiniones'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': [],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }   
an6_fil, an6_agg = analize_wordset_not_so_naive_4(odf, wsw6, True)

busca_tokens(tokens, ['box'])

wsw7 =  {
            'name':'experiencia de envío/entrega/packaging'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    # GENÉRICO DE ENVÍO
                    {
                    'syn0': [],
                    'syn1': ['shipment_arrived'],
                    #'syn2': ['excellent_condition'],
                    'syn2': [],
                    'nots': []
                    }
                    # PAQUETE RECIBIDO EN MALAS CONDICIONES
                    ,{
                    'syn0': ['shipping_damage'],
                    'syn1': ['broken', 'damaged'],
                    'syn2': ['box', 'package'],
                    'nots': []
                    }
                    #,{
                    #'syn0': ['poor_packaging'],
                    #'syn1': ['package', 'package_arrived'],
                    #'syn2': ['unharmed'],
                    #'nots': []
                    #},
                    ,{
                    'syn0': [],
                    'syn1': ['package', 'box'],
                    'syn2': ['missing'],
                    'nots': []
                    }
                    # GENÉRICO DE PAQUETE Y PACKAGING
                    ,{
                    'syn0': busca_tokens(tokens, ['packaging']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': busca_tokens(tokens, ['packed']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,
                    {
                    'syn0': [],
                    'syn1': busca_tokens(tokens, ['package']),
                    #'syn2': ['seal', 'unsealed', 'factory_sealed'],
                    'syn2': [],
                    'nots': []
                    }
                    # ENVÍO RÁPIDO
                    ,{
                    'syn0': ['timely_shipping', 'speedy_shipping', 'overnight_shipping', 'ships_quickly', 'shipped_fast', 'shipped_quickly', 'shipped_promptly'],
                    'syn1': ['shipping', 'delivery'],
                    'syn2': ['fast', 'quick', 'good'],
                    'nots': []
                    }
                    ,{
                    'syn0': ['shipped_immediately', 'fast_delivery', 'fast_service', 'fast_ship', 'fast_shipping', 'quick_delivery'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': ['arrived_timely', 'arrived_earlier', 'arrived_sooner', 'arrived_promptly'],
                    'syn1': ['arrived'],
                    'syn2': ['fast', 'quickly'],
                    'nots': []
                    }
                    ,{
                    'syn0': [],
                    'syn1': ['shipment_arrived'],
                    'syn2': ['short_time'],
                    'nots': []
                    }
                    # ENVÍO LENTO
                    ,{
                    'syn0': [],
                    'syn1': ['shipping', 'delivery'],
                    'syn2': ['slow', 'while', 'bad'],
                    'nots': []
                    }
                    ]
            }
        }
an7_fil, an7_agg = analize_wordset_not_so_naive_4(odf, wsw7, True)

busca_tokens(tokens, ['seller'])
wsw8 =  {
            'name':'experiencia en devolución/postventa'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': ['help_desk', 'customer_support', 'customer_service', 'tech_support', 'call_customer', 'called_customer'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': busca_tokens(tokens, ['warranty']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': busca_tokens(tokens, ['return']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': busca_tokens(tokens, ['returning']),
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ,{
                    'syn0': ['sent_back', 'send_back', 'sending_back', 'ship_back'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an8_fil, an8_agg = analize_wordset_not_so_naive_4(odf, wsw8, True)

busca_tokens(tokens, ['back'])     


wsw9 =  {
            'name':'se menciona el precio'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': [],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an9_fil, an9_agg = analize_wordset_not_so_naive_4(odf, wsw9, True)

wsw10 = {
            'name':''
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': [],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an10_fil, an10_agg = analize_wordset_not_so_naive_4(odf, wsw10, show=True)
     
wsw11 = {
            'name':''
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': [],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an11_fil, an11_agg = analize_wordset_not_so_naive_4(odf, wsw11, show=True)
      
wsw12 = {
            'name':''
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': [],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an12_fil, an12_agg = analize_wordset_not_so_naive_4(odf, wsw12, show=True)
 
wsw13 = {
            'name':''
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': [],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an13_fil, an13_agg = analize_wordset_not_so_naive_4(odf, wsw13, show=True)

wsw14 = {
            'name':''
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': [],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
an14_fil, an14_agg = analize_wordset_not_so_naive_4(odf, wsw14, show=True)

#5an1_fil, an1_agg = analize_wordset_not_so_naive(odf, wsw1, True)
an1_fil, an1_agg = analize_wordset_not_so_naive_3(odf, wsw1, show=True)
#an1_1 = generate_bow(df=an1_fil, show=True)
#an1_2 = generate_bow_post_analize_main(odf_filtered=an1_fil, wsw=wsw1, show=True)
#an1_2 = generate_bow_not_so_naive_main(odf_filtered=an1_fil, wsw=wsw1, stopwords=get_stopwords('post'), show=True)            
an2_fil, an2_agg = analize_wordset_not_so_naive_3(odf, wsw2, show=True)
an3_fil, an3_agg = analize_wordset_not_so_naive_3(odf, wsw3, True)
#an3_1 = generate_bow(df=an3_fil, show=True)
#an3_2 = generate_bow_not_so_naive_main(odf_filtered=an3_fil, wsw=wsw3, stopwords=get_stopwords('post'), show=True)                    
an4_fil, an4_agg = analize_wordset_not_so_naive_3(odf, wsw4, True)
#an4_1 = generate_bow(df=an4_fil, show=True)
#an4_2 = generate_bow_not_so_naive_main(odf_filtered=an4_fil, wsw=wsw4, stopwords=get_stopwords('post'), show=True)
an5_fil, an5_agg = analize_wordset_not_so_naive_3(odf, wsw5, True)
#an5_1 = generate_bow(df=an5_fil, show=True)
#an5_2 = generate_bow_not_so_naive_main(odf_filtered=an5_fil, wsw=wsw5, stopwords=get_stopwords('post'), show=True)
an6_fil, an6_agg = analize_wordset_not_so_naive_3(odf, wsw6, True)
#an6_1 = generate_bow(df=an6_fil, show=True)
#an6_2 = generate_bow_not_so_naive_main(odf_filtered=an6_fil, wsw=wsw6, stopwords=get_stopwords('post'), show=True)
an7_fil, an7_agg = analize_wordset_not_so_naive_3(odf, wsw7, True)
#an7_1 = generate_bow(df=an7_fil, show=True)
#an7_2 = generate_bow_not_so_naive_main(odf_filtered=an7_fil, wsw=wsw7, stopwords=get_stopwords('post'), show=True)
an8_fil, an8_agg = analize_wordset_not_so_naive_3(odf, wsw8, True)
#an8_1 = generate_bow(df=an8_fil, show=True)
#an8_2 = generate_bow_not_so_naive_main(odf_filtered=an8_fil, wsw=wsw8, stopwords=get_stopwords('post'), show=True)
an9_fil, an9_agg = analize_wordset_not_so_naive_3(odf, wsw9, True)
#an9_1 = generate_bow(df=an9_fil, show=True)
#an9_2 = generate_bow_not_so_naive_main(odf_filtered=an9_fil, wsw=wsw9, stopwords=get_stopwords('post'), show=True)
an10_fil, an10_agg = analize_wordset_not_so_naive_3(odf, wsw10, show=True)
an11_fil, an11_agg = analize_wordset_not_so_naive_3(odf, wsw11, show=True)
an12_fil, an12_agg = analize_wordset_not_so_naive_3(odf, wsw12, show=True)
an13_fil, an13_agg = analize_wordset_not_so_naive_3(odf, wsw13, show=True)
an14_fil, an14_agg = analize_wordset_not_so_naive_3(odf, wsw14, show=True)















# BUSCAR RELACIONES ENTRE DOCUMENTOS RETORNADOS POR VARIOS WORDSETS
lista_resultados_busquedas = []
lista_resultados_busquedas.append({'name':wsw1['name'], 'resultados': an1_fil})
lista_resultados_busquedas.append({'name':wsw2['name'], 'resultados': an2_fil})
lista_resultados_busquedas.append({'name':wsw3['name'], 'resultados': an3_fil})
lista_resultados_busquedas.append({'name':wsw4['name'], 'resultados': an4_fil})
lista_resultados_busquedas.append({'name':wsw5['name'], 'resultados': an5_fil})
lista_resultados_busquedas.append({'name':wsw6['name'], 'resultados': an6_fil})
lista_resultados_busquedas.append({'name':wsw7['name'], 'resultados': an7_fil})
lista_resultados_busquedas.append({'name':wsw8['name'], 'resultados': an8_fil})
lista_resultados_busquedas.append({'name':wsw9['name'], 'resultados': an9_fil})
lista_resultados_busquedas.append({'name':wsw10['name'], 'resultados': an10_fil})
lista_resultados_busquedas.append({'name':wsw11['name'], 'resultados': an11_fil})
lista_resultados_busquedas.append({'name':wsw12['name'], 'resultados': an12_fil})
lista_resultados_busquedas.append({'name':wsw13['name'], 'resultados': an13_fil})
lista_resultados_busquedas.append({'name':wsw14['name'], 'resultados': an14_fil})


matriz_doc_ws, matriz_doc_ws_agg = analize_wordset_occurrences(lista_resultados_busquedas)
wordsets_names = list(matriz_doc_ws.columns)
wordsets_names.remove('total_wordsets')
matriz_doc_ws.groupby(wordsets_names).agg({'total_wordsets':np.size})
matriz_doc_ws_expanded = pd.merge(matriz_doc_ws, odf['rating'], left_index=True, right_index=True, how='inner')
matriz_doc_ws_expanded_agg = matriz_doc_ws_expanded.groupby(wordsets_names).agg({'total_wordsets':[np.size, np.mean], 'rating':[np.median, np.mean, np.std]})


visualize_wordsets_network(matriz_doc_ws_expanded, 'F')





heatmap = matriz_doc_ws_expanded.groupby('rating').sum()[wordsets_names]
heatmap_n = heatmap.div(heatmap.max(axis=0), axis=1)
cmap = sns.light_palette("#000000", as_cmap=True)
sns.heatmap(heatmap_n.transpose(), cmap=cmap)
sns.heatmap(heatmap.transpose(), cmap=cmap)




# VER CUÁNTO PENALIZA CADA TEMA, COMPARANDO CON EL RESTO
# EJ: LAS OPINIONES QUE HABLAN DE EASY INSTALL SON CÓMO RESPECTO A LAS OPINIONES QUE NO HABLAN DE EASY INSTALL






# herramientas para localizar temas
words = generate_bow(df=odf)
bigrams = extract_bigrams_from_bow(words)
words['pos'].unique()
words.groupby('pos').count()


sustantivos = words[words['pos'] == 'sustantivo'][['bigram', 'num_occurrences']]





















matriz_doc_ws_sin_hits = matriz_doc_ws[matriz_doc_ws['total_wordsets'] == 0]
documentos_sin_hits = matriz_doc_ws_sin_hits.index.values.tolist()
odf_sin_hits = odf.loc[documentos_sin_hits]
tokens_sin_hits = generate_bow(odf_sin_hits, classify_pos_b=False)
bigrams_sin_hits = extract_bigrams_from_bow(tokens_sin_hits)





'not_recognize'
'signal_strenght'
get_close_words(odf_sin_hits, 3, 'defective_unit', 8)


busca_tokens(tokens, ['defective'])

wswT =   {
            'name':'correct size'
            ,'wordset': 
            {
                'ands': [],
                'ors' :
                    [
                    {
                    'syn0': ['not_recognize'],
                    'syn1': [],
                    'syn2': [],
                    'nots': []
                    }
                    ]
            }
        }
                
anT_fil, anT_agg = analize_wordset_not_so_naive_4(odf_sin_hits, wswT, show=True)















