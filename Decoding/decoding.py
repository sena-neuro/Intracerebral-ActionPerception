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
import pandas as pd
from itertools import combinations
from collections import defaultdict
from sklearn.model_selection import permutation_test_score
from os import path


def decode_each_lead(lead_data_df,clf=svm.SVC(decision_function_shape='ovo')):

    # Function will return this, we will put this to a bigger dict and convert to pandas
    lead_results_dict = {}

    # Calculate combinations of conditions e.g (Body,Per)...
    conditions = lead_data_df.action_category.unique()
    combs = combinations(conditions, 2)

    # Run classification for each of the combinations e.g. Body vs Per ...
    for combination in list(combs):

        # Index of the trials with labels as first or second element of the combination
        ind1 = lead_data_df["action_category"] == combination[0]
        ind2 = lead_data_df["action_category"] == combination[1]

        # Get the data using these indices ( trials with either one label or the other)
        X = lead_data_df[ind1 | ind2].power.tolist()
        y = lead_data_df[ind1 | ind2].action_category.tolist()

        # Create Cross validation
        cv = KFold(n_splits=2, random_state=None, shuffle=True)
        # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        # Permutation test
        # TODO: change n_permutations to a higher value when running
        score, perm_scores, pvalue = permutation_test_score(
            clf, X, y, scoring="accuracy", cv=cv, n_permutations=10)

        # Save the result to a dictionary to use later
        lead_results_dict[combination] = {"score": score,"pvalue": pvalue}
    return lead_results_dict


# Using lead data map, we can do a classification for each lead
# Input path
# Get names of the subject folders

# Mapping from condition to label TENTATIVE
event_code_to_action_category_map = {
    "1": "BOD",
    "4": "OBJ",
    "7": "PER",
}

# Using lead data map, we can do a classification for each lead Input path
input_path = Path(path.join(Path().resolve().parent, 'Data', 'TF_Analyzed'))
output_path = Path(path.join(Path().resolve().parent, 'Results'))

subject_paths = [x for x in input_path.iterdir() if x.is_dir()]
# Classification results dictionary  items are 'lead': [lead_name1,lead_name2,..,]
#                                               'classification_type': (Body,Per) or  (Body,Obj) or (Per,Obj)
#                                               accuracy: [Acc1,Acc2,...,]
#                                               pval: [pval1, pval2,....,]
classification_results_dict = defaultdict(list)

# n_subjects = len(subject_paths)
file = path.join(output_path,'lead_df.pkl')
if Path(file).exists():
    lead_df = pd.read_pickle(file)
else:
    lead_df = read_data(subject_paths[0])
    lead_df.to_pickle(file)

for lead in lead_df.lead.unique():
    df = lead_df.loc[lead_df['lead'] == lead]
    lead_result_dict = decode_each_lead(df)
    for comb, result in lead_result_dict.items():
        classification_results_dict["lead"].append(lead)
        classification_results_dict["classification_type"].append(comb)
        classification_results_dict["accuracy"].append(result["score"])
        classification_results_dict["pvalue"].append(result["pvalue"])

classification_results_file =  path.join(output_path,'classification_results_df.pkl')
classification_results_df = pd.DataFrame.from_dict(classification_results_dict)
classification_results_df.to_pickle(classification_results_file)