import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from os import listdir
import warnings
import json
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer, PorterStemmer

def readfile (path, filename):
    f = open(path + filename, "r")
    txt = f.read().splitlines()

    flag = False
    content = ""
    try:
        label = filename.split("_")[0]
        _id = filename.split("_")[2].split('.')[0]
    except:
        label = 0
        _id = filename.split("_")[1].split('.')[0]

    for line in txt:
        if line[-1:] == ":" : 
            flag = True
            continue
        if flag:
            content += (" " + line)
            
    return (content, label, _id)

def files_to_df(path, files):
    df = pd.DataFrame(columns = ["content", "label", "id"])
    for file in files:
        content, label, _id = readfile(path, file)
        df = df.append({
            "content": content,
            "label": label,
            "id": int(_id)

        }, ignore_index=True)
    return df

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def find_features_in_words(words, features):
        
    for word in features:
        for i in range(len(words)):
            if words[i].startswith(word): 
                return True, i

    return False, -1

def prediction_by_keyword(df):

    # find keywords and predict
    word_features_1 = ["smok", "toba", "cig"]
    word_features_2 = ["deny", "deni", "non", "not", "negative"]
    word_features_3 = ["quit", "past", "former", "ex"]

    predict_Y =list()

    for i in range(len(df["clean_content"])):
        content = df["clean_content"][i]
        words = word_tokenize(content)
        find_smoke, idx = find_features_in_words(words, word_features_1)
        if find_smoke:
            head_idx = max(0, idx - 6)
            tail_idx = min(idx + 5, len(words)-1)
            find_negative, dump = find_features_in_words(words[head_idx:tail_idx], word_features_2)
            if find_negative:
                predict_Y.append(1)     # NON-SMOKER
            else:
                find_past, dump = find_features_in_words(words[head_idx:tail_idx], word_features_3)
                if find_past:
                    predict_Y.append(2)     # PAST SMOKER
                else:
                    predict_Y.append(3)     # CURRENT SMOKER
        else:
            predict_Y.append(0)     # UNKNOWN

    return predict_Y

def translate_string_label_to_number(df_label):
    y = list()
    for i in range(len(df_label)):
        if df_label[i] == 'UNKNOWN':
            y.append(0)
        elif df_label[i] == 'NON-SMOKER':
            y.append(1)
        elif df_label[i] == 'PAST SMOKER':
            y.append(2)
        elif df_label[i] == 'CURRENT SMOKER':
            y.append(3)
        else:
            y.append(-1)

    return y

def translate_number_label_to_string(y):
    label = list()

    for i in range(len(y)):
        if y[i] == 0:
            label.append('UNKNOWN')
        elif y[i] == 1:
            label.append('NON-SMOKER')
        elif y[i] == 2:
            label.append('PAST SMOKER')
        elif y[i] == 3:
            label.append('CURRENT SMOKER')
        else:
            label.append('NAN')

    return label

if __name__ == '__main__':

    # import data
    traning_files = listdir("./data/Training data/")
    testing_files = listdir("./data/Testing data/")
    df = files_to_df("./data/Training data/", traning_files)
    df_test = files_to_df("./data/Testing data/", testing_files)

    # Text processing    
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    
    # stopwords setting
    stopwords = set(stopwords.words('english'))
    symbols   = ['.', ',', '’', '“', '”', '"', "''",
                "'", '*', '``', '**', '$', '%', '&', '#',
                '-', '--', "''", '""', '?', '!', ':', ';',
                '(', ')', '[', ']', '{', '}', "/"]
    stopwords.update(symbols)

    # tokenize, lemmatize, stemming, filter with stopwords
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    clean_content = []

    for i in range(0, len(df["content"])):
        content = df["content"][i]
        word_list = word_tokenize(content.lower())
    # no using stopwords because it will also ignore negative words
    #     lemmatizer_stemming_content = " ".join(
    #         stemmer.stem(lemmatizer.lemmatize(word)) for word in word_list if word not in stopwords
    #     )
        lemmatizer_stemming_content = " ".join(
            stemmer.stem(lemmatizer.lemmatize(word)) for word in word_list
        )
        clean_content.append(lemmatizer_stemming_content)
    df = df.assign(clean_content = clean_content)
    df = df.sort_values(by = ['id'])
    df = df.reset_index(drop = True)

    clean_content = []

    for i in range(0, len(df_test["content"])):
        content = df_test["content"][i]
        word_list = word_tokenize(content.lower())
    # no using stopwords because it will also ignore negative words
    #     lemmatizer_stemming_content = " ".join(
    #         stemmer.stem(lemmatizer.lemmatize(word)) for word in word_list if word not in stopwords
    #     )
        lemmatizer_stemming_content = " ".join(
            stemmer.stem(lemmatizer.lemmatize(word)) for word in word_list
        )
        clean_content.append(lemmatizer_stemming_content)

    df_test = df_test.assign(clean_content = clean_content)
    df_test = df_test.sort_values(by = ['id'])
    df_test = df_test.reset_index(drop = True)
    
    Y = translate_string_label_to_number(df["label"])
    predict_Y = prediction_by_keyword(df)
    
    # result analysis - training set
    hit = 0
    for i in range(len(Y)):
        if Y[i] == predict_Y[i]:
            hit = hit +1
    accuracy = hit / len(Y) * 100
    print('The prediction accuracy is :' + str(accuracy))
    target_names = [ 'UNKNOWN', 'NON-SMOKER', 'PAST SMOKER', 'CURRENT SMOKER']
    plt.figure(figsize=(6,6))
    cnf_matrix = confusion_matrix(Y, predict_Y)
    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=False,
                        title='normalized confusion matrix')

    plt.show()

    # result output - test sets
    predict_Y = prediction_by_keyword(df_test)
    predict_label = translate_number_label_to_string(predict_Y)
    for i in range(len(df_test['id'])):
        print(str(df_test['id'][i]) + ','+predict_label[i])