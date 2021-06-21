#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:42:59 2021

@author: Sena Er github: sena-neuro
"""
from sklearn import svm
from Decoding.read_data import read_data
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, RepeatedKFold, cross_val_score
import pandas as pd
from itertools import combinations
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import multiprocessing as mp
import numpy as np

# Mapping from condition to label
event_code_to_action_category_map = {
    "1": "Skin-Displacing",
    "4": "Manipulative",
    "7": "Interpersonal",
}
action_categories = [ac for ac in event_code_to_action_category_map.values()]
classification_results_list = []

SUBJECT_NAME_INDEX = 0
LEAD_INDEX = 1
ACTION_CLASS_INDEX = 2
POWER_IDX = 3


def binary_action_category_decoder(x, y, cv, clf, grid_search=False, param_grid=None):
    """
    Pairwise action category decoding

    :rtype: float, array (shape: permutation), float
    :returns score, perm_scores, p_value
    :keyword x : Any
        power
    :keyword y : Any
        action category
    """

    if grid_search:
        # Split into training and testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

        search = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, n_jobs=1)
        search.fit(x_train, y_train)
        clf = search.best_estimator_
        x = x_test
        y = y_test

    # Permutation test
    score, perm_scores, p_value = permutation_test_score(
        clf, x, y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

    return score, perm_scores, p_value


def decode_lead(power_arr, cv, clf, grid_search=False, param_grid=None):
    """
    Pairwise action category decoding on each lead

    :returns list
        Stores the results of the classification

    :keyword data_array (nd array)
        Data belonging to a subject
    """

    lead_decoding_results = []

    # Run classification for each of the combinations e.g. Body vs Per ...
    for ac_pair in combinations(action_categories, 2):
        # Indices of the trials belonging to the first or second action category
        # Index 2 is the index of action category
        ind1 = power_arr[:, ACTION_CLASS_INDEX] == ac_pair[0]
        ind2 = power_arr[:, ACTION_CLASS_INDEX] == ac_pair[1]

        only_ac_pair_arr = power_arr[ind1 | ind2]

        # Get the data using the indices (trials with either one label or the other)
        x = only_ac_pair_arr[:, POWER_IDX].tolist()             # power
        y = only_ac_pair_arr[:, ACTION_CLASS_INDEX].tolist()    # action class

        score, _, p_value = binary_action_category_decoder(x, y, cv, clf, grid_search, param_grid)

        subject = power_arr[0, SUBJECT_NAME_INDEX]
        lead = power_arr[0, LEAD_INDEX]
        res = [subject,
               lead,
               ac_pair,
               score,
               p_value,
               len(x)  # number of trials
               ]

        lead_decoding_results.append(res)
    return lead_decoding_results


def log_result(result):
    # This is called whenever decode_each_lead returns a result.
    # result_list is modified only by the main process, not the pool workers.
    classification_results_list.extend(result)
    print("Lead ", result[0][1], " is finished!")


def mp_decode(entire_data_arr):
    pool = mp.Pool(mp.cpu_count()-1)

    # Classifier for the decoding
    clf = make_pipeline(StandardScaler(), svm.SVC())

    # Parameter grid for grid search
    param_grid = {'C': [0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1]}

    # Create Cross validation
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    # Efficiency: May be too slow and not necessary
    # cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)

    # We do decoding on each lead of each subject INDEPENDENTLY
    # They can run at the same time; thus, we use multiprocessing
    # We use pool because they can run asynchronously

    all_unique_subjects = np.unique(entire_data_arr[:, SUBJECT_NAME_INDEX])
    all_unique_leads = np.unique(entire_data_arr[:, LEAD_INDEX])
    for subject_no, subject in enumerate(all_unique_subjects):
        for lead_no, lead in enumerate(all_unique_leads):
            subj_idx = entire_data_arr[:, SUBJECT_NAME_INDEX] == subject
            lead_idx = entire_data_arr[:, LEAD_INDEX] == lead
            single_lead_single_subject_arr = entire_data_arr[subj_idx & lead_idx]
            pool.apply_async(decode_lead,
                             args=(single_lead_single_subject_arr, cv, clf, True, param_grid),
                             callback=log_result)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # Using lead data map, we can do a classification for each lead Input path
    parent_path = Path().resolve().parent
    input_path = parent_path / 'Data' / 'TF_Analyzed'
    output_path = parent_path / 'Results'

    # For server the input and output paths will
    subject_paths = [x for x in input_path.iterdir() if x.is_dir()]

    out_file = output_path / 'lead_df.pkl'
    if Path(out_file).exists():
        lead_df = pd.read_pickle(out_file)
    else:
        lead_df = read_data(subject_paths[0])
        lead_df.to_pickle(out_file)

    lead_arr = lead_df.to_numpy()
    mp_decode(lead_arr)

    classification_results_df = pd.DataFrame.from_records(classification_results_list,
                                                          columns=["Subject",
                                                                   "Lead",
                                                                   "Classification_type",
                                                                   "Accuracy",
                                                                   "P_value",
                                                                   "N_trials"
                                                                   ])
    classification_results_file = output_path / 'classification_results_df_scale_svm.pkl'
    classification_results_df.to_pickle(str(classification_results_file))
