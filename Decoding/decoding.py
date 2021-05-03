#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:42:59 2021

@author: senaer
"""
from Decoding.read_data import read_data
from pathlib import Path
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Using lead data map, we can do a classification for each lead
# Input path
input_path = Path().resolve().parent / 'Data' / 'TF_Analyzed'

# Get names of the subject folders
subject_paths = [x for x in input_path.iterdir() if x.is_dir()]
n_subjects = len(subject_paths)

lead_dict = read_data(subject_paths[0])

# Classifier is LDA
clf = LinearDiscriminantAnalysis()

# Decode action categories on each lead
for lead, data in lead_dict.items():
    X = data['power']
    y = data['action_category']
    for ind, el in enumerate(X):
        X[ind] = np.squeeze(el)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Evaluate model
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # Report performance
    print('Classification score in lead %s: %.3f (%.3f)' % (lead, np.mean(scores), np.std(scores)))
