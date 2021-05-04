#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:42:59 2021

@author: Sena Er github: sena-neuro
"""
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, f1_score, make_scorer

from Decoding.read_data import read_data
from pathlib import Path
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd


# Mapping from condition to label TENTATIVE
event_code_to_action_category_map = {
    "1": "BOD",
    "4": "OBJ",
    "7": "PER",
}


def body_accuracy_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=["BOD", "OBJ", "PER"], normalize="true" ) # normalize can be True not sure
    return cm.diagonal()[0]

def object_accuracy_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=["BOD", "OBJ", "PER"], normalize="true" ) # normalize can be True not sure
    return cm.diagonal()[1]


def person_accuracy_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=["BOD", "OBJ", "PER"], normalize="true" ) # normalize can be True not sure
    return cm.diagonal()[2]


# Decode action categories on each lead
def decode_each_lead(power, action_category,
                     clf=svm.SVC(decision_function_shape='ovo')):
    X = power.tolist()
    y = action_category.tolist()

    for ind, el in enumerate(X):
        X[ind] = np.squeeze(el)
        y[ind] = event_code_to_action_category_map[y[ind]]

    # Cross validation
    cv = KFold(n_splits=2, random_state=None, shuffle=False)
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # pairwise_f1_scorer = make_scorer(f1_score, labels=["BOD", "OBJ", "PER"], average=None)
    # Evaluate model
    body_scores = cross_val_score(clf, X, y, scoring=body_accuracy_scorer, cv=cv, n_jobs=-1)
    object_scores = cross_val_score(clf, X, y, scoring=object_accuracy_scorer, cv=cv, n_jobs=-1)
    person_scores = cross_val_score(clf, X, y, scoring=person_accuracy_scorer, cv=cv, n_jobs=-1)
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    # Report performance
    print('----------------------------')
    print('---------- ',lead,' ---------')
    print('Body accuracy for lead {}: {:.3f} ({:.3f})'.format(lead, np.mean(body_scores), np.std(body_scores)))
    print('Object accuracy for lead {}: {:.3f} ({:.3f})'.format(lead, np.mean(object_scores), np.std(object_scores)))
    print('Person accuracy for lead {}: {:.3f} ({:.3f})'.format(lead, np.mean(person_scores), np.std(person_scores)))

    print('Overall accuracy in lead {}: {:.3f} ({:.3f})'.format(lead, np.mean(scores), np.std(scores)))
    print('----------------------------')
# Using lead data map, we can do a classification for each lead
# Input path
input_path = Path().resolve().parent / 'Data' / 'TF_Analyzed'

# Get names of the subject folders
subject_paths = [x for x in input_path.iterdir() if x.is_dir()]
# n_subjects = len(subject_paths)

file = 'temp.pkl'
if Path(file).exists():
    lead_df = pd.read_pickle(file)
else:
    lead_df = read_data(subject_paths[0])
    lead_df.to_pickle(file)


for lead in lead_df.lead.unique():
    X = lead_df.loc[lead_df['lead'] == lead].power
    y = lead_df.loc[lead_df['lead'] == lead].action_category

    decode_each_lead(X, y)
