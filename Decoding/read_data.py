import re
import numpy as np
import scipy.io
import scipy.io
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
from itertools import combinations


# TODO: HOW TO TEST IF LEAD CONDITON ETC MATCHES
def read_data(subject_path):
    # Mapping from condition to label TENTATIVE
    event_code_to_action_category_map = {
        "1": "BOD",
        "4": "OBJ",
        "7": "PER",
    }

    # A dictinary to keep datapoint and labels associated with each lead
    # Each key is the name of the lead and each value is a dictionary
    # with data and labels. These lists have the same orders (labels are matching the data).
    lead_dict = defaultdict(list)

    # Get names of the condition files in the subjecets folder
    condition_paths = [x for x in subject_path.iterdir() if x.match('*.mat')]
    n_conditions = len(condition_paths)

    # Read each condition file and create label
    for condition_path in condition_paths:

        condition_mat_struct = scipy.io.loadmat(condition_path, struct_as_record=False)["D"]
        n_leads = len(condition_mat_struct)

        pattern = "condition\_(.*?)\_"
        event_code = re.search(pattern, condition_path.stem).group(1)

        # Find the indices of underscore character
        # The condition label is between the 2nd and 3rd underscores
        # Maybe get only the used part of the label but we can still use only a part of this in next steps
        action_category = event_code[0]

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

            # Check if the lead is iEEG There are some probable unnecesssary channels here e.g EOG2 DEL4?
            # If the AR_tfX has a size 0f 0 then all trials must have been rejected.
            if lead_data.ChanType[0] == "iEEG" and lead_data.AR_tfX.size != 0:

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
                    # since 1 is Body, 4 is OBj and 7 is Person
                    labels.extend([action_category] * no_trials)
                else:
                    # If there is only one trial then the shape will be 2d,
                    # add it as one element to the list
                    data_vectors.append(power.flatten())
                    labels.append(action_category)

                # Save the leads data and labels in a map
                lead_dict["lead"].extend([lead_name]*no_trials)
                lead_dict["action_category"].extend(labels)
                lead_dict["power"].extend(data_vectors)

    lead_df = pd.DataFrame.from_dict(lead_dict)
    return lead_df
