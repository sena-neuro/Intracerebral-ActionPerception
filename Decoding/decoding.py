#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:42:59 2021

@author: senaer
"""

#import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

# IO
a_file = open("lead_dict.pkl", "rb")
lead_dict = pickle.load(a_file)

# Classifier is LDA
clf = LinearDiscriminantAnalysis()

# Decode action categories on each lead
for lead, data in lead_dict.items():
    X = data['power']
    y = data['action_category']

    clf.fit(X, y)