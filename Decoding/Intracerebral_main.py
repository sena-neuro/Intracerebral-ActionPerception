#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 13:46:23 2021

@author: senaer
"""

import scipy.io
import re
from pathlib import Path
import pickle
from collections import defaultdict
# import pandas

# Mapping from condition to label TENTATIVE
event_code_to_action_category_map = {
    "1":"BOD",
    "4":"OBJ",
    "7":"PER",
}

input_path = Path().resolve().parent / 'Data' / 'TF_Analyzed'

# df = pd.DataFrame(columns=['Subject', 'Lead', 'Action Category', 'Power'])

# A dictinary to keep datapoint and labels associated with each lead
# Each key is the name of the lead and each value is a dictionary 
# with data and labels. These lists have the same orders (labels are matching the data).
lead_dict = defaultdict(lambda: defaultdict(list))

####
subject_paths = [x for x in input_path.iterdir() if x.is_dir()]
n_subjects = len(subject_paths)
# TODO Iterate over  multiple subjects -> for


# Get names of the condition files in the subjects folder
# TODO do we need to do it seperately?
condition_paths = [x for x in subject_paths[0].iterdir() if x.match('*.mat')]

# Read each condition file and create label
for c in condition_paths:
    
    mat_struct = scipy.io.loadmat(c, struct_as_record=False)["D"]
    n_leads = len(mat_struct)

    pattern = "condition\_(.*?)\_"
    event_code = re.search(pattern, c.stem).group(1)
    # We can also use the conditions first character here for the class
    # since 1 is Body, 4 is OBj and 7 is Person
    action_category = event_code_to_action_category_map[event_code[0]]

    # We will do the classification for each lead, run a loop for all leads
    # We need to scan all files of a subject to be able to start classification since different files
    # include different conditions
    for idx, lead in enumerate(mat_struct):

        # Get the data in the lead
        lead_data = lead[0][0][0]

        # Pass/ignore not iEEG leads and leads with no trials (all trials are rejected)
        # If size of the AR_tfX is 0 then all trials were rejected
        if  lead_data.ChanType[0] != "iEEG" or lead_data.AR_tfX.size == 0:
            continue

        # Get lead name
        lead_name = lead_data.ChanName[0]

        # Get the power information (50x200xtrial)
        # 50: number of frequencies
        # 200: number of timepoints
        # Number of trials change between 1 and 4
        power = lead_data.AR_power
        
        for i in range(power.shape[-1]):
            lead_dict[lead_name]["power"].append(power[...,i].flatten())
            lead_dict[lead_name]["action_category"].append(action_category)
 
# defaultdict -> dict for pickling
# May translate to Pandas DataFrame in the future
for value in lead_dict:
    lead_dict[value] = dict(lead_dict[value])
lead_dict = dict(lead_dict)  
   
# Using lead data map, we can do a classification for each lead
# Save lead_data_map
a_file = open("lead_dict.pkl", "wb")
pickle.dump(lead_dict, a_file)
a_file.close()
