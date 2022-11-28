import pandas as pd
import numpy as np

from unidecode import unidecode
from string import punctuation
from re import sub
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt

from modelling import run_modelling


""" --------- 1. Data Collection --- """

def drop_unnecessary_columns(data, columns_needed):
    columns_all = list(data.columns)
    columns_not_needed = []

    for col in columns_all:
        if col not in columns_needed:
            columns_not_needed.append(col)
    
    data.drop(columns_not_needed, axis=1, inplace=True)

    return data

def label_cleaning(data):
    # clean spaces and lower all letters
    data['Category'] = data['Category'].str.strip()
    data['Category'] = data['Category'].str.lower()

    data['Abstract'] = data['Abstract'].apply(lambda x: text_cleaning(x))

    return data

def text_cleaning(x):

    punctuation_table = {}
    for char in set(punctuation):
        punctuation_table[char] = " "

    x = sub('\s+', ' ', unidecode(x).translate(x.maketrans(punctuation_table)).lower())
    return x

""" --------- 2. Data Analysis ----- """

def count_words(data):

    cat_stat = pd.DataFrame(data['Category'].value_counts().sort_index())
    cat_stat.columns=['No_Of_Papers']

    # percentage of papers in each category
    cat_stat['%_Of_Papers'] = round(cat_stat['No_Of_Papers']/data.shape[0]*100, 0)
    print(cat_stat)
    

    cat_stat['No_Of_Words'] = 0
    cat_stat['Words_String'] = None
    cat_stat['Words_List'] = None

    for cat in data['Category'].unique():
        all_words_str = ''
        for idx, row in enumerate(data['Category']):
            if cat == row:
                all_words_str = all_words_str + data['Abstract'][idx]
        
        cat_stat.at[cat, 'No_Of_Words'] = len(all_words_str.split(' '))
        cat_stat.at[cat, 'Words_String'] = all_words_str
        # tokenization
        cat_stat.at[cat, 'Words_List'] = all_words_str.split(' ') 

    cat_stat['No_Of_Words_per_paper'] = round(cat_stat['No_Of_Words']/cat_stat['No_Of_Papers'], 0)

    # for save the avg_words_plot
    plt.figure(figsize=(8,8))
    plt.title('avg words plot')
    avg_words_plot = cat_stat['No_Of_Words_per_paper'].plot(kind='bar', rot=90, legend=False)
    avg_words_plot.set(xlabel="Category", ylabel="Count", title="Average number of words in papers for each category")
    plt.savefig("imgs/avg_words_plot.png")


    return cat_stat

""" --------- 3. Featuring --------- """

""" 3.1. Text Processing """

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

def count_words_cleaned(cat_stat, data):

    cat_stat['No_Of_Words_Cleaned'] = 0
    cat_stat['Words_String_Cleaned'] = None
    cat_stat['Words_List_Cleaned'] = None

    for cat in data['Category'].unique():
        all_words_str = ''
        for idx, row in enumerate(data['Category']):
            if cat == row:
                all_words_str = all_words_str + data['Words_S'][idx]
        
        cat_stat.at[cat, 'No_Of_Words_Cleaned'] = len(all_words_str.split(' '))
        cat_stat.at[cat, 'Words_String_Cleaned'] = all_words_str
        cat_stat.at[cat, 'Words_List_Cleaned'] = all_words_str.split(' ')
    
    return cat_stat

""" 3.2. Labelling With Numbers """

def create_categories_dict(data):

    categories = list(data['Category'].unique())
    categories_enum = enumerate(categories, 0)
    categories_dict_1 = dict(categories_enum)
    categories_dict_2 = {}

    for k, v in categories_dict_1.items():
        categories_dict_2[v] = k
    print('Category numbering: ', categories_dict_2)

    return categories_dict_1, categories_dict_2

def categories_to_nums(data, categories_dict_2):
    data["Y_cat"] = data["Category"].map(categories_dict_2)

    return data

def set_proper_types_of_columns(data, columns_needed):
    cat_dict = {'Y_cat': 'category', 
                'Keywords': 'str',
                'Abstract': 'str',
                'Category': 'str'}
    
    for col in columns_needed:
        data[col] = data[col].astype(cat_dict[col])
    
    return data

def run():
    # loading dataset into pandas dataframe
    # data columns -> Y1	Y2	Y	Category	area	keywords	Abstract

    data = pd.read_excel('data/Data.xlsx')
    # for control
    print(data.head(3))

    data.columns = ['Y1', 'Y2', 'Y', 'Category', 'area', 'keywords', 'Abstract']

    print('Shape:\n' , data.shape)     # rows x columns
    print('Columns:\n' , data.columns) # column names

    data.info(memory_usage="deep")

    columns_needed = ['Category', 'Abstract']

    data = drop_unnecessary_columns(data=data, columns_needed=columns_needed)
    # for control
    print(data.head(3))

    data.info(verbose=False, memory_usage="deep")

    data['Category'].unique()
    data = label_cleaning(data=data)

    cat_stat = count_words(data=data)

    data = create_words(data=data)
    # for control
    print(data.head(3))

    cat_stat = count_words_cleaned(cat_stat=cat_stat, data=data)
    categories_dict_1, categories_dict_2 = create_categories_dict(data=data)
    data = categories_to_nums(data=data, categories_dict_2=categories_dict_2)
    # for control
    print(data.head(5))
    data = set_proper_types_of_columns(data=data, columns_needed=columns_needed)    

    run_modelling(data=data, categories_dict_1=categories_dict_1, cat_stat=cat_stat)

if __name__=='__main__':
    run()