#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:42:59 2021

@author: Sena Er github: sena-neuro
"""
from sklearn import svm
from Decoding.read_data import read_data
from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold
import pandas as pd
from itertools import combinations
from collections import defaultdict
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Mapping from condition to label TENTATIVE
event_code_to_action_category_map = {
    "1": "Body-Displacing",
    "4": "Manipulative",
    "7": "Interpersonal",
}


def decode_each_lead(lead_data_df, clf):
    """
    Pairwise action category decoding on each lead

    :returns lead_results_dict (Pandas Dataframe)
        Stores the results of the classification

    :keyword lead_data_df (Pandas Dataframe)
        Data belonging to a subject
    """

    # Function will return this, we will put this to a bigger dict and convert to pandas
    lead_results_dict = {}

    # Calculate all pair combinations between the 3 action categories
    # e.g Manipulative vs Interpersonal
    action_categories = lead_data_df.action_category.unique()
    action_category_pairs = combinations(action_categories, 2)

    # Run classification for each of the combinations e.g. Body vs Per ...
    for ac_pair in list(action_category_pairs):
        # Indices of the trials belonging to the first or second action category
        ind1 = lead_data_df["action_category"] == ac_pair[0]
        ind2 = lead_data_df["action_category"] == ac_pair[1]

        # Get the data using the indices (trials with either one label or the other)
        x = lead_data_df[ind1 | ind2].power.tolist()
        y = lead_data_df[ind1 | ind2].action_category.tolist()

        # Create Cross validation
        # cv = KFold(n_splits=5, random_state=0, shuffle=True)
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)

        # Permutation test
        # TODO: change n_permutations to a higher value when running on server
        score, perm_scores, p_value = permutation_test_score(
            clf, x, y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=-1)

        # Save the result to a dictionary to use later
        lead_results_dict[ac_pair] = {"score": score, "p_value": p_value, "n_trials": len(x)}
    return lead_results_dict


# Using lead data map, we can do a classification for each lead Input path
parent_path = Path().resolve().parent
input_path = parent_path / 'Data' / 'TF_Analyzed'
output_path = parent_path / 'Results'

# For server the input and output paths will
subject_paths = [x for x in input_path.iterdir() if x.is_dir()]
# Classification results dictionary  items are 'lead': [lead_name1,lead_name2,..,]
#                                               'classification_type': (Body,Per) or  (Body,Obj) or (Per,Obj)
#                                               accuracy: [Acc1,Acc2,...,]
#                                               pval: [pval1, pval2,....,]
classification_results_dict = defaultdict(list)

# n_subjects = len(subject_paths)
out_file = output_path / 'lead_df.pkl'
if Path(out_file).exists():
    lead_df = pd.read_pickle(out_file)
else:
    lead_df = read_data(subject_paths[0])
    lead_df.to_pickle(out_file)

# Classifier for the decoding
clf = make_pipeline(StandardScaler(), svm.SVC())

for subject_no, subject in enumerate(lead_df.subject_name.unique()):
    for lead_no, lead in enumerate(lead_df.lead.unique()):

        # Not necessary after updating the lead df file
        print("----Processing Lead number:  ", lead_no, "of ", len(lead_df.lead.unique()), "---------")

        df = lead_df.loc[(lead_df['lead'] == lead) &
                         (lead_df['subject_name'] == subject)]

        # lead_result_dict = decode_each_lead(df)
        # to use scaling
        # lead_result_dict = decode_each_lead(df, clf=make_pipeline(StandardScaler(), svm.SVC()))
        # To use LDA
        lead_result_dict = decode_each_lead(df, clf=clf)
        for comb, result in lead_result_dict.items():
            classification_results_dict["subject"].append(subject)
            classification_results_dict["lead"].append(lead)
            classification_results_dict["classification_type"].append(comb)
            classification_results_dict["accuracy"].append(result["score"])
            classification_results_dict["p_value"].append(result["p_value"])
            classification_results_dict["n_trials"].append(result["n_trials"])

classification_results_file = output_path / 'classification_results_df_scale_LDA.pkl'
classification_results_df = pd.DataFrame.from_dict(classification_results_dict)
classification_results_df.to_pickle(classification_results_file)
