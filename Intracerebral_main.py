import os
import scipy.io
import numpy as np

# Input path
inPath='/Users/huseyinelmas/CCNLAB/IntacerebralEEG_ActionBase/TF_Analyzed/'


# Mapping from condition to label TENTATIVE
label_condition_map = {
    "1":"BOD",
    "4":"OBJ",
    "7":"PER",
}

# A dictinary to keep datapoint and labels associated with each lead
# Each key is the name of the lead and each value is a dictionary 
# with data and labels. These lists have the same orders (labels are matching the data).
lead_data_map = {}



# Get names of the subject folders
subject_folders = [name for name in os.listdir(inPath) if
                   os.path.isdir(inPath + name) and not name.startswith(".")]
n_subjects = len(subject_folders)

## We will do the classification for each subject
subject_path = inPath+subject_folders[0]+"/"

# Get names of the condition files in the subjecets folder
condition_files = [name for name in os.listdir(subject_path) if name.endswith(".mat")]
n_conditions = len(condition_files)

# Read each condition file and create label
for condition_file in condition_files:
    condition_mat_file = scipy.io.loadmat(subject_path + "/"+ condition_file, struct_as_record=False)["D"]

    # Find the indices of underscore character
    us_idx = [i for i, ltr in enumerate(condition_file) if ltr == "_"]

    # The condition label is between the 2nd and 3rd underscores
    # Maybe get only the used part of the label but we can still use only a part of this in next steps
    cond = condition_file[us_idx[1]+1: us_idx[2]]

    # We will do the classification for each lead, run a loop for all leads
    # We need to scan all files of a subject to be able to start classification since different files
    # include different conditions
    for lead_idx in range(len(condition_mat_file)):

        # Data points
        X = []

        # Labels
        Y = []

        # Get the data in the lead
        lead_data = condition_mat_file[lead_idx][0][0][0]

        # Get lead name
        lead_name = lead_data.ChanName[0]

        # Check if the lead is iEEG There are some probable unnecesssary channels here e.g EOG2 DEL4?
        # If the AR_tfX has a size 0f 0 then all trials must have been rejected.
        if lead_data.ChanType[0] == "iEEG" and lead_data.AR_tfX.size != 0:

            # Get the power information (50x200xtrial)
            power = lead_data.AR_power
            if len(power.shape) == 3:

            	# Add the power to the data points list
            	# If there iss more than one trial left 
            	# then add the trials as list elements
                X.extend(np.dsplit(power,power.shape[2]))
                
                # Add condition as label and add it as many times as the power has trials
            	# We can also use the conditions first character here for the class
            	# since 1 is Body, 4 is OBj and 7 is Person
                Y.extend([label_condition_map[cond[0]]] * power.shape[2])

            else:

                # If there is only one trial then the shape will be 2d,
                # add it as one element to the list
                X.append(power)
                Y.append(label_condition_map[cond[0]])
               

            # Save the leads data and labels in a map
            if lead_name not in lead_data_map.keys():
                lead_data_map[lead_name] = {"data": X, "labels": Y}
            else:
                lead_data_map[lead_name]["data"].extend(X)
                lead_data_map[lead_name]["labels"].extend(Y)
print("done")
# Using lead data map, we can do a classification for each lead

