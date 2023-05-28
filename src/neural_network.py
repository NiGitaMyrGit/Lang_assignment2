#!/usr/bin/env python3
# before running this script be sure to run bash setup.sh in the command line inside the main directory
# help: nbs 5
## BEGIN ###

# system tools
import os
import sys
sys.path.append(".")

import joblib

# data munging tools
import pandas as pd
# importing Ross' classifier utils
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import classification_report
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt

# importing argument parser
# TODO find out how to make real argparse
import argparse
from argparse import ArgumentParser
parser = ArgumentParser()

parser = argparse.ArgumentParser(description='neural network model')

def load_data():
    # loading in the data
    filename = os.path.join("in", "fake_or_real_news.csv")
    # load into dataframe
    data = pd.read_csv(filename, index_col=0)
    #creating new variables, taking data out of dataframe
    X = data["text"]
    y = data["label"]
    # initial preprocessing, train test split
    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                        y,               
                                                        test_size=0.2,   
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def vectorize_data(X_train, X_test):
    # vectorize data  
    vectorizer = TfidfVectorizer(ngram_range = (1,2), 
                                lowercase =  True,
                                max_df = 0.95,
                                min_df = 0.05,
                                max_features = 500)

    # fit to training data... 
    X_train_feats = vectorizer.fit_transform(X_train)
    # fit to test data
    X_test_feats = vectorizer.transform(X_test)
    return vectorizer, X_train_feats, X_test_feats


def MLP_model(X_train_feats, X_test_feats, y_train):
    classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20,),
                           max_iter=1000,
                           random_state = 42)
    classifier.fit(X_train_feats, y_train)
    #get predictions
    y_pred = classifier.predict(X_test_feats)
    return y_pred, classifier

#Classifier report
def classifier_report(y_test, y_pred, classifier, vectorizer):
    #Saving classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    report_path = os.path.join("out", "NN_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(classifier_metrics)
    # save the trained models and vectorizers
    from joblib import dump, load
    # classifier and vecotrizer output paths
    classifier_path = os.path.join("models", "NN_model.joblib")
    vectorizer_path = os.path.join("models", "TF-IDF_vectorizer.joblib")
    # save the model
    dump(classifier, classifier_path)
    # save the vectorizer
    dump(vectorizer, vectorizer_path)

#main function
def main():
    #load data
    X_train, X_test, y_train, y_test = load_data()
    #vectorize
    vectorizer, X_train_feats, X_test_feats = vectorize_data(X_train, X_test)
    # run classifier
    y_pred, classifier = MLP_model(X_train_feats, X_test_feats, y_train)
    # save model
    classifier_report(y_test, y_pred, classifier, vectorizer)
    

# calling main function
if __name__== "__main__":
    main()
