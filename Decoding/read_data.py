"""

@author: Sena Er (github: sena-neuro) & Huseyin Orkun Elmas
"""
import h5py
import scipy.io
import re
from pathlib import Path

# Mapping from condition to label
event_code_to_action_category_map = {
    "1": "SD",  # Skin-displacing
    "4": "MN",  # Manipulative
    "7": "IP",  # Interpersonal
}

# Using lead data map, we can do a classification for each lead Input path
parent_path = Path('/auto/data2/oelmas/Intracerebral')
input_path = parent_path / 'Data' / 'TF_Analyzed'
output_path = parent_path / 'Data'

power_hdf_file = str(output_path / 'power_data.hdf5')


def mat_to_hdf(subject_path : Path):
    """

    """

    # Create and/or open HDF5 in append mode
    file = h5py.File(power_hdf_file, 'a')
    subject_name = subject_path.stem
    
    if subject_name in file.keys(): # Not tested as of 13 October 2021
        print(subject_name, " is already in the HDF file. Moving on to the next subject...")
    else:
        condition_paths = [x for x in subject_path.iterdir() if x.match('*.mat')]

        for condition_path in condition_paths:
            condition_mat_struct = scipy.io.loadmat(condition_path, struct_as_record=False)["D"]
            # print(condition_path.stem)
            ac_code = re.search(r"_\d\d\d", condition_path.stem).group()[1]
            action_category = event_code_to_action_category_map[ac_code]

            apostrophe = re.compile("(')")

            for idx, lead in enumerate(condition_mat_struct):

                # Get the data in the lead
                lead_data = lead[0][0][0]
                # Get lead name
                lead_name = lead_data.ChanName[0]
                # Check if the lead name follows the format and isnt one of the unnecessary leads
                lead_name_correct = re.match("([A-Z]'?(([0-9]+'?)|[A-Z]?)$)", lead_name)


                # Check if the lead is iEEG There are some probable unnecesssary channels here e.g EOG2 DEL4?
                # If the AR_tfX has a size 0f 0 then all trials must have been rejected.
                if lead_data.ChanType[0] == "iEEG" and lead_name_correct and lead_data.AR_tfX.size != 0:

                    # Get the power information
                    power = lead_data.AR_power  # (50, 200, n_trial) = (n_freqs, n_times, n_trials)
                    # times = lead_data.AR_times[0]  # (200,)
                    # freqs = lead_data.AR_freqs[0]  # (50,)

                    # We can't have names with apostrophes so
                    # change the apostrophe with an underscore
                    lead_name_no_ap = apostrophe.sub('_', lead_name)

                    # Create a HDF5 group with the subject and the lead
                    subject_lead_key = subject_name + '/' + lead_name_no_ap
                    # Open the group if it is not created before create it first
                    group = file.require_group(subject_lead_key)
                    
                    power = power.reshape(power.shape[0], power.shape[1], -1) # (n_freqs, n_times, n_trials)
                    n_trials = power.shape[2]
                    
                    for time_idx in range(200):
                        # Take all freqs and trials for a specific time point
                        power_t = power[:, time_idx, :].transpose()    
                        
                        time_ac_key = 't_' + str(time_idx) + '/' + action_category

                        # The key is not in the group keys
                        # First time the key is formed
                        # Dataset is created
                        if time_ac_key not in group.keys():
                            group.create_dataset(time_ac_key,
                                                 data=power_t,
                                                 maxshape=(64, 50)
                                                 )

                        else:
                            dataset = group[time_ac_key]
                            dataset.resize((dataset.shape[1] + n_trials, 50))
                            dataset[-n_trials:, :] = power_t

        print(subject_name, " is processed. Moving on to the next subject...")
    file.close()


def process_each_subject():
    subject_paths = [x for x in input_path.iterdir() if x.is_dir()]

    for s_path in subject_paths:
        mat_to_hdf(s_path)


if __name__ == '__main__':
    process_each_subject()
