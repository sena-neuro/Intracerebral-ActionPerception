"""

@author: Sena Er (github: sena-neuro) & Huseyin Orkun Elmas
"""
import numpy as np
import scipy.io
import re
from collections import defaultdict
import pandas as pd
from pathlib import Path
import multiprocessing as mp

# Mapping from condition to label TENTATIVE
event_code_to_action_category_map = {
    "1": "Skin-Displacing",
    "4": "Manipulative",
    "7": "Interpersonal",
}

all_subjects_dict = defaultdict(list)

def read_data(subject_path):
    """
    Reads the mat files (each belonging to a condition) inside the subject's folder.
    Makes a Pandas DataFrame (lead_df) belonging to the subject

    :returns lead_df
        lead_df has the columns: subject_name, lead, action_category, power

    :keyword subject_path -- Path of the subject's folder
    """

    # lead_dict: A dictionary to keep datapoint and labels associated with each lead
    #   Each key is the name of the lead and each value is a dictionary with data and labels.
    #   These lists have the same orders (labels are matching the data).
    lead_dict = defaultdict(list)

    # Get the names of the condition files in the subjects folder

    condition_paths = [x for x in subject_path.iterdir() if x.match('*.mat')]

    # Read each condition file and create label
    for condition_path in condition_paths:

        condition_mat_struct = scipy.io.loadmat(condition_path, struct_as_record=False)["D"]
        # n_leads = len(condition_mat_struct)
        # Get the codes written in the subject file name and seperated with _

        code_list = condition_path.stem.split("_")
        event_code = code_list[2]
        subject_name = code_list[0]

        # Find the indices of underscore character
        # The condition label is between the 2nd and 3rd underscores
        # Maybe get only the used part of the label but we can still use only a part of this in next steps
        action_category = event_code_to_action_category_map[event_code[0]]

        # We will do the classification for each lead, run a loop for all leads
        # We need to scan all files of a subject to be able to start classification since different files
        # include different conditions
        for idx, lead in enumerate(condition_mat_struct):

            # Data points
            data_vectors = []

            # Labels
            labels = []

            # Get the data in the lead
            lead_data = lead[0][0][0]

            # Get lead name
            lead_name = lead_data.ChanName[0]

            # Check if the lead name follows the format and isnt one of the unnecessary leads
            lead_name_correct = re.match("([A-Z]'?(([0-9]+'?)|[A-Z]?)$)", lead_name)

            # Check if the lead is iEEG There are some probable unnecesssary channels here e.g EOG2 DEL4?
            # If the AR_tfX has a size 0f 0 then all trials must have been rejected.
            if lead_data.ChanType[0] == "iEEG" and lead_data.AR_tfX.size != 0 and lead_name_correct:

                # Get the power information (50x200xtrial)
                power = lead_data.AR_power

                # Get number rof trials, if number of trials is one then tte power array will have 2 dims
                no_trials = power.shape[2] if len(power.shape) == 3 else 1

                # If there are more than one trials, convert each trial to an element of the list
                if no_trials > 1:

                    no_trials = power.shape[2]
                    # Add the power to the data points list
                    # If there is more than one trial left
                    # then add the trials as list elements
                    temp = np.hsplit(power.reshape(-1, power.shape[-1]), no_trials)

                    # Squeeze each trial
                    for i in range(len(temp)):
                        temp[i] = temp[i].squeeze()

                    data_vectors.extend(temp)
                    # Add condition as label and add it as many times as the power has trials
                    # We can also use the conditions first character here for the class
                    # since 1 is Skin-Displacing, 4 is Manipulative and 7 is Interpersonal
                    labels.extend([action_category] * no_trials)
                else:
                    # If there is only one trial then the shape will be 2d,
                    # add it as one element to the list
                    data_vectors.append(power.flatten())
                    labels.append(action_category)

                # Save the leads data and labels in a map
                lead_dict["subject_name"].extend([subject_name] * no_trials)
                lead_dict["lead"].extend([lead_name] * no_trials)
                lead_dict["action_category"].extend(labels)
                lead_dict["power"].extend(data_vectors)

    return lead_dict


def log_result(subject_dict):
    # This is called whenever read_data returns a result.
    # all_subjects_dict is modified only by the main process, not the pool workers.
    all_subjects_dict["subject_name"].extend(subject_dict["subject_name"])
    all_subjects_dict["lead"].extend(subject_dict["lead"])
    all_subjects_dict["action_category"].extend(subject_dict["action_category"])
    all_subjects_dict["power"].extend(subject_dict["power"])


if __name__ == '__main__':

    # Using lead data map, we can do a classification for each lead Input path
    parent_path = Path('/auto/data2/oelmas/Intracerebral')
    input_path = parent_path / 'Data' / 'TF_Analyzed'
    output_path = parent_path / 'Results'

    # For server the input and output paths will
    subject_paths = [x for x in input_path.iterdir() if x.is_dir()]

    out_file = output_path / 'all_subjects_data.pkl'

    pool = mp.Pool(mp.cpu_count() - 1)

    for s_path in subject_paths:
        pool.apply_async(read_data, args=(s_path, ), callback=log_result)

    pool.close()
    pool.join()

    all_subjects_df = pd.DataFrame.from_dict(all_subjects_dict)
    all_subjects_df.to_pickle(out_file)
