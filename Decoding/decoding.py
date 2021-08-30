#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:42:59 2021

@author: Sena Er github: sena-neuro
"""
from sklearn import svm
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
from itertools import combinations
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import multiprocessing as mp
import numpy as np
import datetime

# Mapping from condition to label
event_code_to_action_category_map = {
    "1": "Skin-Displacing",
    "4": "Manipulative",
    "7": "Interpersonal",
}
action_categories = [ac for ac in event_code_to_action_category_map.values()]
current_subject_decoding_results_list = []
current_subject = None

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
        search = GridSearchCV(clf,
                              param_grid=param_grid,
                              cv=cv,
                              n_jobs=1)
        search.fit(x, y)
        clf = search.best_estimator_
        params = search.best_params_
    else:
        # I didn't really checked this but I don't think it would break something
        params = clf.get_params()

    # Permutation test
    score, perm_scores, p_value = permutation_test_score(
        clf, x, y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

    return score, p_value, params


def decode_lead(power_arr, cv, clf, grid_search=False, param_grid=None):
    """
    Pairwise action category decoding on each lead

    :returns list
        Stores the results of the classification

    :keyword power_arr (nd array)
        Data belonging to a subject
    """
    lead_decoding_results = [None, None, None]
    subject = power_arr[0, SUBJECT_NAME_INDEX]
    lead = power_arr[0, LEAD_INDEX]

    ac_pair_idx = 0
    # Run classification for each of the combinations e.g. Body vs Per ...
    for ac_pair in combinations(action_categories, 2):
        # Indices of the trials belonging to the first or second action category
        # Index 2 is the index of action category
        ind1 = power_arr[:, ACTION_CLASS_INDEX] == ac_pair[0]
        ind2 = power_arr[:, ACTION_CLASS_INDEX] == ac_pair[1]

        only_ac_pair_arr = power_arr[ind1 | ind2]

        # Get the data using the indices (trials with either one label or the other)
        x = only_ac_pair_arr[:, POWER_IDX].tolist()  # power
        y = only_ac_pair_arr[:, ACTION_CLASS_INDEX].tolist()  # action class

        score, p_value, params = binary_action_category_decoder(x, y, cv, clf, grid_search, param_grid)
        param_C = params['clf__C']

        lead_decoding_results[ac_pair_idx] = [subject,
                                              lead,
                                              ac_pair,
                                              score,
                                              p_value,
                                              clf,
                                              param_C,
                                              len(x)  # number of trials
                                              ]
        ac_pair_idx = ac_pair_idx + 1

    return lead_decoding_results


def log_result(result):
    # This is called whenever decode_each_lead returns a result.
    # result_list is modified only by the main process, not the pool workers.
    current_subject_decoding_results_list.extend(result)


def mp_decode(power_arr):
    pool = mp.Pool(mp.cpu_count() - 1)

    # Classifier for the decoding
    clf = Pipeline([('scale', StandardScaler()),
                    ('clf', svm.SVC(kernel='linear'))])

    # Parameter grid for grid search
    param_grid = {'clf__C': [2.5e-05, 5e-05, 7.5e-05, 1e-04, 2.5e-04, 5e-04]}

    gs = True

    # Create Cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    # We do decoding on each lead of each subject INDEPENDENTLY
    # They can run at the same time; thus, we use multiprocessing
    # We use pool because they can run asynchronously

    all_unique_leads = np.unique(power_arr[:, LEAD_INDEX])
    for lead_no, lead in enumerate(all_unique_leads):
        lead_idx = power_arr[:, LEAD_INDEX] == lead
        single_lead_arr = power_arr[lead_idx]
        pool.apply_async(decode_lead,
                         args=(single_lead_arr, cv, clf, gs, param_grid),
                         callback=log_result)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # Using lead data map, we can do a classification for each lead Input path
    parent_path = Path('/auto/data2/oelmas/Intracerebral')
    input_path = parent_path / 'Data' / 'Power_DataFrames'
    output_path = parent_path / 'Results' / 'SubjectDecodingResults'

    date = datetime.datetime.today().strftime('%d-%m')
    subject_pkl = [x for x in input_path.iterdir() if x.match('*.pkl')]
    out_pkls = [x for x in output_path.iterdir() if x.match('*.pkl')]

    for s_pkl in subject_pkl:
        subject_df = pd.read_pickle(s_pkl)
        subject_arr = subject_df.to_numpy()

        # global current_subject
        current_subject = subject_arr[0, SUBJECT_NAME_INDEX]

        # If the subject is decoded before on any date do not decode it again
        pattern = '*' + current_subject + '*'
        if not any(out_pkl.match(pattern) for out_pkl in out_pkls):
            mp_decode(subject_arr)
            subject_decoding_results_df = pd.DataFrame.from_records(current_subject_decoding_results_list,
                                                                    columns=["subject",
                                                                             "lead",
                                                                             "classification_type",
                                                                             "accuracy",
                                                                             "p_value",
                                                                             "clf",
                                                                             "param_C",
                                                                             "n_trials"
                                                                             ])
            current_subject_decoding_results_list = []

            file_name = date + '_' + current_subject + '_decoding_results.pkl'
            subject_decoding_results_pkl = str(output_path / file_name)
            subject_decoding_results_df.to_pickle(subject_decoding_results_pkl)
