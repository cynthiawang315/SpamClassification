#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:03:58 2019

@author: cynthiawang
"""

#%%
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#%%
# load the file, tokenize, remove stopwords, lemmatize, create a dataframe
with open('SpamCollection.txt','r') as messages:
    the_messages = messages.readlines()

import spacy
# run this in terminal: python -m spacy download en
sp = spacy.load('en')
message_df = pd.DataFrame() 
stop_words = stopwords.words('english')
for message in the_messages:
    label = re.search("(ham)|(spam)", message).group()
    body = re.sub("((ham)|(spam))\s*","", message)
    body = re.sub('[^A-z0-9]+',' ',body)

    cleaned = [word.lower() for word in word_tokenize(body) if word not in stop_words]
    joined = " ".join(cleaned)
    sentence = sp(joined)
    lemmatized = [word.lemma_ for word in sentence]

    message_df = message_df.append({"body": " ".join(lemmatized), "label": label}, ignore_index = True)


#%%

#split data    
y = message_df.label
X_train, X_test, y_train, y_test = train_test_split(message_df, y, test_size=0.2, random_state=42)

#%%
# TfidfVectorizer
my_vec_tfidf = TfidfVectorizer()
my_xform_tfidf = my_vec_tfidf.fit_transform(X_train.body).toarray()
col_names_tfidf = my_vec_tfidf.get_feature_names()
my_xform_tfidf = pd.DataFrame(my_xform_tfidf, columns = col_names_tfidf)

# CountVectorizer
my_vec_count = CountVectorizer()
my_xform_count = my_vec_count.fit_transform(X_train.body).toarray()
col_names_count = my_vec_count.get_feature_names()
my_xform_count = pd.DataFrame(my_xform_count, columns = col_names_count)


#%%
# determine n_components for PCA
    
import matplotlib.pyplot as plt
def find_n_components(my_vec, desired_variance):
    
    my_pca = PCA().fit(my_vec)
    plt.figure()
    variance_sum = np.cumsum(my_pca.explained_variance_ratio_)
    plt.plot(np.cumsum(my_pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Explained Variance')
    pca_graph = plt.show()
    
    for index in range(len(variance_sum)):
        if variance_sum[index] >= desired_variance:
            break;
    return index, pca_graph

# PCA with n_components
index, pca_graph = find_n_components(my_xform_tfidf,0.95)    
pca = PCA(n_components = index)
my_dim_tfidf = pca.fit_transform(my_xform_tfidf)



############ end of pre-processing ##############


#%%
# grid search - random forest
from sklearn.ensemble import RandomForestClassifier
def grid_search_func(param_grid, the_model_in, the_vec_in, the_label_in):
    grid_search = GridSearchCV(the_model_in, param_grid=param_grid, cv=5)
    best_model = grid_search.fit(the_vec_in, the_label_in)
    max_score = grid_search.best_score_
    best_params = grid_search.best_params_

    return best_model, max_score, best_params

clf_pca = RandomForestClassifier()
param_grid = {"max_depth": [10, 50, 100],
              "n_estimators": [1, 4, 16, 32, 64],
              "random_state": [1234]}

gridsearch_model, best, opt_params = grid_search_func(param_grid, clf_pca, my_dim_tfidf, y_train)

#%%
# training - random forest
clf_pca.set_params(**gridsearch_model.best_params_)
clf_pca.fit(my_dim_tfidf, y_train)

#%%
# performance measure - random forest

def fetch_prediction(data_in):
    the_predicted_out = pd.DataFrame()
    for word, actual in zip(data_in.body,data_in.label):
        test_text = my_vec_tfidf.transform([word]).toarray() 
        test_text_pca = pca.transform(test_text) 
        the_probs = list(clf_pca.predict_proba(test_text_pca)[0])
        the_predict = clf_pca.predict(test_text_pca)[0] 
        the_probs.append(the_predict)
        the_probs.append(actual)
        the_result = pd.DataFrame(the_probs).T 
        tmp_cols =np.append(clf_pca.classes_, 'predicted') 
        the_result.columns = np.append(tmp_cols, 'actual')
        the_predicted_out = the_predicted_out.append(the_result, ignore_index=True)

    return the_predicted_out

prediction_df = fetch_prediction(X_test)

from sklearn.metrics import precision_recall_fscore_support
accuracy = float(list(
        (prediction_df.actual == prediction_df.predicted)
        ).count(True))/float(len(prediction_df))

the_measures = pd.DataFrame(
        precision_recall_fscore_support(prediction_df.actual,
                                        prediction_df.predicted,labels=["ham","spam"])).T
# 0:ham ; 1:spam
the_measures.columns = ['precision', 'recall', 'fscore', 'accuracy']
the_measures.accuracy = accuracy
print (the_measures)
cmtx_rf = pd.DataFrame(
    confusion_matrix(prediction_df.actual, prediction_df.predicted, labels = ["ham","spam"]), 
    index=['true:ham', 'true:spam'], 
    columns=['pred:ham', 'pred:spam']
)
print(cmtx_rf)
#%%
# grid search - logistic regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
param_grid_log = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}
gridsearch_model_log, best_log, opt_params_log = grid_search_func(param_grid_log, logreg, my_dim_tfidf, y_train)

#%%
# training - logistic regression
logreg.set_params(**gridsearch_model_log.best_params_)
logreg.fit(my_dim_tfidf, y_train)

#%%
# performance measure - logistic regression
def fetch_prediction_log(data_in):
    the_predicted_out = pd.DataFrame()
    for word, actual in zip(data_in.body,data_in.label):
        test_text = my_vec_tfidf.transform([word]).toarray() 
        test_text_pca = pca.transform(test_text) 
        the_probs = list(logreg.predict_proba(test_text_pca)[0]) 
        the_predict = logreg.predict(test_text_pca)[0]
        the_probs.append(the_predict)
        the_probs.append(actual)
        the_result = pd.DataFrame(the_probs).T 
        tmp_cols =np.append(logreg.classes_, 'predicted') 
        the_result.columns = np.append(tmp_cols, 'actual')
        the_predicted_out = the_predicted_out.append(the_result, ignore_index=True)

    return the_predicted_out

prediction_df_log = fetch_prediction_log(X_test)

accuracy_log = float(list(
        (prediction_df_log.actual == prediction_df_log.predicted)
        ).count(True))/float(len(prediction_df_log))

the_measures_log = pd.DataFrame(
        precision_recall_fscore_support(prediction_df_log.actual,
                                        prediction_df_log.predicted,labels=["ham","spam"])).T
# 0:ham ; 1:spam
the_measures_log.columns = ['precision', 'recall', 'fscore', 'accuracy']
the_measures_log.accuracy = accuracy_log
print (the_measures_log)

cmtx_log = pd.DataFrame(
    confusion_matrix(prediction_df_log.actual, prediction_df_log.predicted, labels = ["ham","spam"]), 
    index=['true:ham', 'true:spam'], 
    columns=['pred:ham', 'pred:spam']
)
print(cmtx_log)
#%%
# training - naive bayes   (no PCA applied)
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(my_xform_tfidf, y_train)

#%%
# performance measure - naive bayes   (no PCA applied)

def fetch_prediction_mnb(data_in):
    the_predicted_out = pd.DataFrame()
    for word, actual in zip(data_in.body,data_in.label):
        test_text = my_vec_tfidf.transform([word]).toarray() 
#        test_text_pca = pca.transform(test_text)
#        test_text_minmax = min_max_scaler.transform(test_text_pca)
        the_probs = list(mnb.predict_proba(test_text)[0]) 
        the_predict = mnb.predict(test_text)[0]
        the_probs.append(the_predict)
        the_probs.append(actual)
        the_result = pd.DataFrame(the_probs).T 
        tmp_cols =np.append(mnb.classes_, 'predicted') 
        the_result.columns = np.append(tmp_cols, 'actual')
        the_predicted_out = the_predicted_out.append(the_result, ignore_index=True)

    return the_predicted_out

prediction_df_mnb = fetch_prediction_mnb(X_test)

from sklearn.model_selection import cross_val_score
cv_mnb = sum(cross_val_score(mnb, my_xform_tfidf, y_train, cv=5))/5

accuracy_mnb = float(list(
        (prediction_df_mnb.actual == prediction_df_mnb.predicted)
        ).count(True))/float(len(prediction_df_mnb))

the_measures_mnb = pd.DataFrame(
        precision_recall_fscore_support(prediction_df_mnb.actual,
                                        prediction_df_mnb.predicted,labels=["ham","spam"])).T
# 0:ham ; 1:spam
the_measures_mnb.columns = ['precision', 'recall', 'fscore', 'accuracy']
the_measures_mnb.accuracy = accuracy_mnb
print (the_measures_mnb)

cmtx_mnb = pd.DataFrame(
    confusion_matrix(prediction_df_mnb.actual, prediction_df_mnb.predicted, labels = ["ham","spam"]), 
    index=['true:ham', 'true:spam'], 
    columns=['pred:ham', 'pred:spam']
)
print(cmtx_mnb)
#%%
# grid search - SVM
from sklearn.svm import SVC
clf_svm = SVC(probability=True)
param_grid_svm = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['rbf']}  
gridsearch_model_svm, best_svm, opt_params_svm = grid_search_func(param_grid_svm, clf_svm, my_dim_tfidf, y_train)

#%%
# training model - SVM
clf_svm.set_params(**gridsearch_model_svm.best_params_)
clf_svm.fit(my_dim_tfidf, y_train)

#%%
# performance measure - SVM

def fetch_prediction_svm(data_in):
    the_predicted_out = pd.DataFrame()
    for word, actual in zip(data_in.body,data_in.label):
        test_text = my_vec_tfidf.transform([word]).toarray() 
        test_text_pca = pca.transform(test_text) 
        the_probs = list(clf_svm.predict_proba(test_text_pca)[0]) 
        the_predict = clf_svm.predict(test_text_pca)[0]
        the_probs.append(the_predict)
        the_probs.append(actual)
        the_result = pd.DataFrame(the_probs).T 
        tmp_cols =np.append(clf_svm.classes_, 'predicted') 
        the_result.columns = np.append(tmp_cols, 'actual')
        the_predicted_out = the_predicted_out.append(the_result, ignore_index=True)

    return the_predicted_out

prediction_df_svm = fetch_prediction_svm(X_test)

accuracy_svm = float(list(
        (prediction_df_svm.actual == prediction_df_svm.predicted)
        ).count(True))/float(len(prediction_df_svm))

the_measures_svm = pd.DataFrame(
        precision_recall_fscore_support(prediction_df_svm.actual,
                                        prediction_df_svm.predicted,labels=["ham","spam"])).T
# 0:ham ; 1:spam
the_measures_svm.columns = ['precision', 'recall', 'fscore', 'accuracy']
the_measures_svm.accuracy = accuracy_svm
print (the_measures_svm)

cmtx_svm = pd.DataFrame(
    confusion_matrix(prediction_df_svm.actual, prediction_df_svm.predicted, labels = ["ham","spam"]), 
    index=['true:ham', 'true:spam'], 
    columns=['pred:ham', 'pred:spam']
)
print(cmtx_svm)