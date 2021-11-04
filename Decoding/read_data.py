"""

@author: Sena Er (github: sena-neuro) & Huseyin Orkun Elmas
"""
import h5py
import scipy.io
import re
from pathlib import Path
import numpy as np
import pickle

# Mapping from condition to label
event_code_to_action_category_map = {
    "1": "SD",  # Skin-displacing
    "4": "MN",  # Manipulative
    "7": "IP",  # Interpersonal
}

if Path().owner() == 'senaer':
    parent_path = Path('/Users/senaer/Codes/CCNLab/Intracerebral-ActionPerception')
else:
    parent_path = Path('/auto/data2/oelmas/Intracerebral')
input_path = parent_path / 'Data' / 'TF_Analyzed'
output_path = parent_path / 'Data'

hdf_file = output_path / 'intracerebral_action_data.hdf5'

lead_name_to_idx = dict()
next_lead_idx = 0

subject_names_set = {'BarellS', 'BenedettiL', 'BerberiM', 'BonsoE', 'CioffiML',
                     'DeianaF', 'DelGiorgioF', 'DobratoV', 'FarciM', 'FazekasD',
                     'GhiniJ', 'GuigliF', 'LoriA', 'MarchesiniS2', 'MarrandinoS',
                     'MeringoloD', 'PriviteraM', 'RollaS', 'SalimbeniR', 'SelvaBoninoR',
                     'TrombacciaR', 'VentroneS', 'VeugelersD', 'VivianB', 'LodiG',
                     'FedericoR', 'FraniM', 'MoniF', 'MilitaruR', 'MorandiniS',
                     'DeProprisA', 'PetriceanuC', 'SavaP', 'RomanoA', 'OttoboniV'
                     'PlutinoA', 'ProneA', 'MartiniML', 'WairN', 'ZanoniM'}


def process_subject(subject_path: Path, verbose=True):
    """

    """
    s_name_beg = re.compile(subject_path.stem[:4])
    subject_name = next(filter(s_name_beg.match, subject_names_set))

    condition_paths = [x for x in subject_path.iterdir() if x.match('*.mat')]

    for condition_path in condition_paths:
        condition_mat_struct = scipy.io.loadmat(condition_path, struct_as_record=False)["D"]

        condition_code = re.search(r"\d\d\d", condition_path.stem).group()
        ac_code = condition_code[0]
        action_category = event_code_to_action_category_map[ac_code]
        trial_code_1 = int(condition_code[1])
        trial_code_2 = int(condition_code[2])

        dset = hdf[action_category + '/power']

        for idx, lead in enumerate(condition_mat_struct):
            # Get the data in the lead
            lead_data = lead[0][0][0]
            # Get lead name
            channel_name = lead_data.ChanName[0]

            # Check if the lead name follows the format and it is not one of the unnecessary leads
            # such as c4, DEL1, EOG1, pz, E, MILO2, TIB1, p3, p4,
            # Starts with a capital letter, apostrophe may follow, ends with digits (at least one digit)
            is_a_lead_of_interest = re.match(r"^([A-Z]'?\d+)$", channel_name)

            # Check if the lead is iEEG
            # Check if the lead is a wanted lead
            # Check if the AR_tfX has a size 0 (=all trials were rejected)
            # Skip the lead if any condition holds
            if lead_data.ChanType[0] == "iEEG" and is_a_lead_of_interest and lead_data.AR_tfX.size != 0:
                subj_lead_name = construct_lead_name(subject_name, channel_name)

                if verbose:
                    # Report the progress
                    print(subj_lead_name, condition_code)

                if subj_lead_name not in lead_name_to_idx.keys():
                    global next_lead_idx
                    lead_name_to_idx[subj_lead_name] = next_lead_idx
                    next_lead_idx = next_lead_idx + 1

                current_lead_idx = lead_name_to_idx[subj_lead_name]

                if subj_lead_name in dset.attrs:
                    dset.attrs[subj_lead_name] = dset.regionref[current_lead_idx, ...]

                # Import power data, swap the time and frequency axes, i.e., (50, 200, n_trials) -> (200, 50, n_trials)
                # Make it c-contiguous, add trials axis if non-existent (2d->3d)
                power = np.atleast_3d(np.ascontiguousarray(np.swapaxes(lead_data.AR_power, 0, 1)))
                # times = lead_data.AR_times[0]  # (200,)
                # freqs = lead_data.AR_freqs[0]  # (50,)
                n_trials = power.shape[2]

                # Compute trial indices
                # trial_code_1 and trial_code_2 both go from 1 to 4
                trials_begin_idx = (((trial_code_1 - 1) * 4) + trial_code_2 - 1) * 4
                trials_end_idx = trials_begin_idx + n_trials
                dset.write_direct(power, dest_sel=np.s_[current_lead_idx, ..., trials_begin_idx:trials_end_idx])


def construct_lead_name(subject_name, lead_name):
    lead_code = re.compile(r"([A-Z])('?)(\d+)")
    channel_letter, apostrophe, channel_number = lead_code.match(lead_name).groups()
    if len(channel_number) == 1:
        channel_number = "0" + channel_number

    if apostrophe == "'":
        hemisphere = 'Left'
    else:
        hemisphere = 'Right'

    subject_lead_name = hemisphere + '_' + subject_name + '_' + channel_letter + '_' + channel_number
    return subject_lead_name


if __name__ == '__main__':
    if hdf_file.exists():
        hdf_file.unlink()

    # Create and/or open HDF5 in append mode
    hdf = h5py.File(hdf_file, 'a')

    # Create and open datasets
    MN_power_dset = hdf.create_dataset('MN/power', (5000, 200, 50, 64), fillvalue=-1, dtype=np.float32)
    SD_power_dset = hdf.create_dataset('SD/power', (5000, 200, 50, 64), fillvalue=-1, dtype=np.float32)
    IP_power_dset = hdf.create_dataset('IP/power', (5000, 200, 50, 64), fillvalue=-1, dtype=np.float32)

    subject_paths = [x for x in input_path.iterdir() if x.is_dir()]
    # Process each subject
    for s_path in subject_paths:
        # futureTODO check if the subject name is in the subject name indices dictionary?
        # If it is pass the subject
        #    print(subject_name, " is already in the HDF file. Moving on to the next subject...")
        process_subject(s_path)

    # Trim necessarily
    MN_power_dset.resize((next_lead_idx, 200, 50, 64))
    SD_power_dset.resize((next_lead_idx, 200, 50, 64))
    IP_power_dset.resize((next_lead_idx, 200, 50, 64))

    hdf.attrs['lead_name_to_idx'] = pickle.dumps(lead_name_to_idx, protocol=0)
    # Usage
    # lead_name_to_idx = pickle.loads(hdf.attrs['lead_name_to_idx'])
