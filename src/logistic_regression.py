#!/usr/bin/env python3

# system tools
import os
import sys
sys.path.append(".")
import joblib

# data munging tools
import pandas as pd
import numpy as np
# importing Ross' classifier utils
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import classification_report
from sklearn import metrics


# importing argument parser
# TODO make real argparse arguments
import argparse
from argparse import ArgumentParser
parser = ArgumentParser()
parser = argparse.ArgumentParser(description='logistic regression model')

#reading in filename

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
    vectorizer = CountVectorizer(ngram_range = (1,2), 
                                lowercase =  True,
                                max_df = 0.95,
                                min_df = 0.05,
                                max_features = 100)

    # fit to training data... 
    X_train_feats = vectorizer.fit_transform(X_train)
    # fit to test data
    X_test_feats = vectorizer.transform(X_test)
    return vectorizer, X_train_feats, X_test_feats


def logreg_model(X_train_feats, y_train, X_test_feats):
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    #get predictions
    y_pred = classifier.predict(X_test_feats)
    return y_pred, classifier

#Classifier report
def classifier_report(y_test, y_pred, classifier, vectorizer):
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    report_path = os.path.join("out", "log_reg_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(classifier_metrics)
    # save the trained models and vectorizers
    from joblib import dump, load
    # output paths for classifier and vectorizer
    classifier_path = os.path.join("models", "Logreg_model.joblib")
    vectorizer_path = os.path.join("models", "Count_vectorizer.joblib")
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
    y_pred, classifier = logreg_model(X_train_feats, y_train, X_test_feats)
    #Saving classification report
    classifier_report(y_test, y_pred, classifier, vectorizer)
    

# calling main function
if __name__== "__main__":
    main()
