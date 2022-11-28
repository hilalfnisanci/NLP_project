import pandas as pd
import numpy as np
import pickle
from unidecode import unidecode
from string import punctuation
from re import sub

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_data(pickle_path):
    with open(pickle_path, 'rb') as handle:
        data = pickle.load(handle)
    
    return data

""" data collection """
def set_types_of_columns(data, columns_needed=['Abstract']):
    cat_dict = {'Abstract' : 'str'}

    for col in columns_needed:
        data[col] = data[col].astype(cat_dict[col])
    
    return data

""" data cleaning """
def abstract_cleaning(data):
    data['Abstract_processed'] = data['Abstract'].apply(lambda x: text_cleaning(x))
    return data

def text_cleaning(x):
    punctuation_table = {}

    for char in set(punctuation):
        punctuation_table[char] = " "
    
    x = sub('\s+', ' ', unidecode(x).translate(x.maketrans(punctuation_table)).lower())
    return x

""" featuring """
def create_words(data):
    """
    Words_L -> words list
    Words_S -> words string
    """

    data['Words_L']=data['Abstract'].apply(lambda words_list: create_text(words_list))
    data['Words_L']=data['Words_L'].apply(lambda words_list: leaving_out_words(words_list))
    data['Words_S']=data['Words_L'].apply(lambda words_list: join_words(words_list))

    return data

def create_text(words_list):

    words_list = words_list.split(' ')
    words_list = [w.strip() for w in words_list if w.isalpha()]
    
    return words_list

def leaving_out_words(words_list):
    # removing stopwords
    stopwords_set = set(stopwords.words('english'))
    words_list = [w for  w in words_list if w not in stopwords_set]
    # lemmatization
    lemma = WordNetLemmatizer()
    words_list = [lemma.lemmatize(w) for w in words_list]

    return words_list

def join_words(words_list):
    # Creating 1 string from list
    return " ".join(words_list)

""" modelling """

def create_vector(X, vectorizer):
    X_vector = vectorizer.transform(X)
    return X_vector

def predict_cat(model, X_vector):
    pred = model.predict(X_vector)
    return pred

""" finalization"""

def data_predicted_xlsx(data, pred, categories_dict_1):
    predicted_data_path = 'data/data_predicted.xlsx'
    data['Predicted_Category'] = pred
    data['Predicted_Category'] = data['Predicted_Category'].map(categories_dict_1)
    data.drop(['Abstract_processed', 'Words_L', 'Words_S'], axis=1, inplace=True)
    data.to_excel(predicted_data_path)

    return data

def test():
    data_path = 'data/test_data.xlsx'
    data = pd.read_excel(data_path)

    data = set_types_of_columns(data=data)
    data = abstract_cleaning(data=data)
    data = create_words(data=data)

    pickle_path = 'exports/tfidf_vectorizer.pkl'
    tfidf_vectorizer = load_data(pickle_path=pickle_path)

    pickle_path = 'exports/model_MNB.pkl'
    model_MNB = load_data(pickle_path=pickle_path)

    pickle_path = 'exports/categories_dict_1.pkl'
    categories_dict_1 = load_data(pickle_path=pickle_path)

    X = data['Words_S']

    X_vector = create_vector(X=X, vectorizer=tfidf_vectorizer)
    
    pred = predict_cat(model=model_MNB, X_vector=X_vector)
    data = data_predicted_xlsx(data=data, pred=pred, categories_dict_1=categories_dict_1)

if __name__ == '__main__':
    test()