from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pickle

from tabulate import tabulate

def train_test(X, y):
    """
    X -> features
    y -> labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size=0.2, random_state=4)
    
    return X_train, X_test, y_train, y_test

"""
functions for one-hot encoding 
"""

def create_vectorizer(analyzer='word', preprocessor=None, max_features=None, min_df=3, 
                    max_df=0.9, ngram_range=(1, 2)):

    tfidf_vectorizer = TfidfVectorizer(analyzer=analyzer,
                        preprocessor=preprocessor,
                        max_features=max_features,
                        min_df=min_df,
                        max_df=max_df,
                        ngram_range=ngram_range)
    
    return tfidf_vectorizer

def create_vector(vectorizer, X_train):

    vectorizer.fit(np.array(X_train))
    X_train_vector = vectorizer.transform(X_train)

    return X_train_vector

"""
functions for time delay
"""
def start_timestamp():
    # saves the starting time

    ts1 = datetime.now()
    return ts1

def end_timestamp(ts1):
    # counts the delay from the starting time

    ts2 = datetime.now()
    delay = ts2-ts1

    return delay

# metrics and predictions

def create_predictions_dataframe(y_test, categories_dict_1):

    predictions = pd.DataFrame(y_test)
    predictions['Category'] = predictions['Y_cat'].map(categories_dict_1)
    return predictions

def add_to_predictions(model_name, pred, predictions, categories_dict_1):

    if not model_name in predictions.columns:
        predictions[model_name] = pred
    model_name_ = str(model_name) + '_'
    predictions[model_name_] = predictions[model_name].map(categories_dict_1)

    return predictions

models_eval_dict = {}
def metrics_evaluation(model, model_name, X_test_vect, y_test, delay, predictions):
    from sklearn import metrics
    if model:
        pred = model.predict(X_test_vect)
    else:
        pred = predictions[model_name]
    acc_score = metrics.accuracy_score(y_test, pred)
    models_eval_dict[model_name] = round(acc_score*100, 1), delay

    return pred, delay, models_eval_dict

# function for model evaluation
def best_model_eva(predictions, model, X_test, y_test, pred=None):
    if pred is not None:
        y_pred = pred
    else:
        y_pred = model.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=predictions["Category"].unique(),
                yticklabels=predictions["Category"].unique())
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin Edilen')
    plt.savefig("imgs/model_evaluation.png")

    print(metrics.classification_report(y_test, y_pred, target_names=predictions["Category"].unique()))

    return y_pred

"""
functions and training ML Models
"""

""" Multinomial Naive Bayes model function """

def model_MultinomialNB(X_train_vect, vectorizer, y_train, X_test, **hyperparam):
    
    from sklearn.naive_bayes import MultinomialNB

    model_name = 'MultinomialNB'
    ts1 = start_timestamp()
    model_MNB = MultinomialNB(alpha=0.1, **hyperparam)
    model_MNB.fit(X_train_vect, y_train)
    
    X_test_vect_MNB = vectorizer.transform(X_test)
    delay = end_timestamp(ts1)
    print("\nDelay: ", delay)
    return model_MNB, delay, X_test_vect_MNB, model_name

def store_data(data, pickle_path):
    # Stores data as a pickle

    with open(pickle_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_modelling(data, categories_dict_1, cat_stat):

    y = data['Y_cat']

    # features for ML model
    X_str = data['Words_S']
    X_str_train, X_str_test, y_train, y_test = train_test(X=X_str, y=y)

    print("X_str_test: \n",X_str_test.head(3))
    
    print("y_test:\n",y_test.head(3))

    tfidf_vectorizer = create_vectorizer()
    X_train_vector = create_vector(vectorizer=tfidf_vectorizer, X_train=X_str_train)

    predictions = create_predictions_dataframe(y_test=y_test, categories_dict_1=categories_dict_1)
    print("predictions:\n",predictions.head(10))

    # Multinomial Naive Bayes model training and evaluation
    model_MNB, delay, X_test_vect_MNB, model_name = model_MultinomialNB(X_train_vect=X_train_vector,
                                        vectorizer=tfidf_vectorizer, y_train=y_train, X_test=X_str_test)
    
    pred, delay, models_eval_dict = metrics_evaluation(model=model_MNB, 
                                                      model_name=model_name,
                                                      X_test_vect=X_test_vect_MNB,
                                                      y_test=y_test, delay=delay,
                                                      predictions=predictions)
                                                      
    predictions = add_to_predictions(model_name=model_name, 
                                    pred=pred, 
                                    predictions=predictions,
                                    categories_dict_1=categories_dict_1)


    models_eval = pd.DataFrame.from_dict(models_eval_dict, columns=['Accuracy', 'Delay'], orient='index')
    
    y_pred = best_model_eva(predictions=predictions, model = None, X_test=None, y_test=y_test, pred=pred)

    pickle_path = 'exports/categories_dict_1.pkl'
    store_data(data=categories_dict_1, pickle_path=pickle_path)

    pickle_path = 'exports/model_MNB.pkl'
    store_data(data=model_MNB, pickle_path=pickle_path)

    pickle_path = 'exports/tfidf_vectorizer.pkl'
    store_data(data=tfidf_vectorizer, pickle_path=pickle_path)
