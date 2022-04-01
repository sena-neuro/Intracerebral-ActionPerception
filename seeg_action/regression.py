from seeg_action import project_config as cfg
import mne
import numpy as np
from mne.stats import linear_regression


def design_matrix_map(design, condition):
    if design == 'action_class_specificity':
        not_working_dict = {

            'MN-ST': [1, -0.5, -0.5, -0.5, 0.25, 0.25, -0.5, 0.25, 0.25],
            'MN-SC': [0, 1, 0, 0, 0, 0, 0, 0, 0],
            'MN-DC': [0, 0, 1, 0, 0, 0, 0, 0, 0],

            'SD-ST': [-0.5, 0.25, 0.25, 1, -0.5, -0.5, -0.5, 0.25, 0.25],
            'SD-SC': [0, 0, 0, 0, 1, 0, 0, 0, 0],
            'SD-DC': [0, 0, 0, 0, 0, 1, 0, 0, 0],

            'IP-ST': [-0.5, 0.25, 0.25, -0.5, 0.25, 0.25, 1, -0.5, -0.5],
            'IP-SC': [0, 0, 0, 0, 0, 0, 0, 1, 0],
            'IP-DC': [0, 0, 0, 0, 0, 0, 0, 0, 1],

        }
        # Works
        working_dict = {

            'MN-ST': [1, 0, 0, -1, 0, 0, -1, 0, 0],
            'MN-SC': [-1, 1, 0, 0, 0, 0, 0, 0, 0],
            'MN-DC': [-1, 0, 1, 0, 0, 0, 0, 0, 0],

            'SD-ST': [-1, 0, 0, 1, 0, 0, -1, 0, 0],
            'SD-SC': [0, 0, 0, -1, 1, 0, 0, 0, 0],
            'SD-DC': [0, 0, 0, -1, 0, 1, 0, 0, 0],

            'IP-ST': [-1, 0, 0, -1, 0, 0, 1, 0, 0],
            'IP-SC': [0, 0, 0, 0, 0, 0, -1, 1, 0],
            'IP-DC': [0, 0, 0, 0, 0, 0, -1, 0, 1],
        }
        # Works
        _dict = {

            'MN-ST': [1, 0, 0, -1, 0, 0, -1, 0, 0],
            'MN-SC': [-1, 1, 0, 1, 0, 0, 1, 0, 0],
            'MN-DC': [-1, 0, 1, 1, 0, 0, 1, 0, 0],

            'SD-ST': [-1, 0, 0, 1, 0, 0, -1, 0, 0],
            'SD-SC': [1, 0, 0, -1, 1, 0, 1, 0, 0],
            'SD-DC': [1, 0, 0, -1, 0, 1, 1, 0, 0],

            'IP-ST': [-1, 0, 0, -1, 0, 0, 1, 0, 0],
            'IP-SC': [1, 0, 0, 1, 0, 0, -1, 1, 0],
            'IP-DC': [1, 0, 0, 1, 0, 0, -1, 0, 1],
        }
    elif design == 'action_activation':
        # Works
        _dict = {
            'MN-ST': [1, -0.5, -0.5, 0, 0, 0, 0, 0, 0],
            'SD-ST': [0, 0, 0, 1, -0.5, -0.5, 0, 0, 0],
            'IP-ST': [0, 0, 0, 0, 0, 0, 1, -0.5, -0.5],
            'MN-SC': [0, 1, 0, 0, 0, 0, 0, 0, 0],
            'SD-SC': [0, 0, 0, 0, 1, 0, 0, 0, 0],
            'IP-SC': [0, 0, 0, 0, 0, 0, 0, 1, 0],
            'MN-DC': [0, 0, 1, 0, 0, 0, 0, 0, 0],
            'SD-DC': [0, 0, 0, 0, 0, 1, 0, 0, 0],
            'IP-DC': [0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
    elif design == 'action_vs_static_control':
        not_working_dict = {
            'MN-ST': [1, -1, 0, 0, 0, 0, 0, 0, 0],
            'SD-ST': [0, 0, 0, 1, -1, 0, 0, 0, 0],
            'IP-ST': [0, 0, 0, 0, 0, 0, 1, -1, 0],
            'MN-SC': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'SD-SC': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IP-SC': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'MN-DC': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'SD-DC': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IP-DC': [0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    elif design == 'action_vs_dynamic_control':
        not_working_dict = {
            'MN-ST': [1, 0, -1, 0, 0, 0, 0, 0, 0],
            'SD-ST': [0, 0, 0, 1, 0, -1, 0, 0, 0],
            'IP-ST': [0, 0, 0, 0, 0, 0, 1, 0, -1],
            'MN-SC': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'SD-SC': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IP-SC': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'MN-DC': [0, 0, 1, 0, 0, 0, 0, 0, 0],
            'SD-DC': [0, 0, 0, 0, 0, 1, 0, 0, 0],
            'IP-DC': [0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
    return _dict[condition]


def regression(epochs, design):
    design_matrix = np.empty(shape=(len(epochs.events), 9))
    for condition in epochs.event_id:
        design_matrix[epochs._keys_to_idx(condition)] = design_matrix_map(design, condition)
    predictor_vars = ['MN-ST', 'MN-SC', 'MN-DC', 'SD-ST', 'SD-SC', 'SD-DC', 'IP-ST', 'IP-SC', 'IP-DC']

    return linear_regression(epochs,
                             design_matrix=design_matrix,
                             names=predictor_vars)


if __name__ == '__main__':
    raw_fif_file = cfg.steps_save_path / 'BerberiM_STEP_0_annotated_raw.fif'
    raw = mne.io.read_raw_fif(raw_fif_file)

    lead = "O'5"
    channel_raw = raw.copy().pick(lead)  # pick channel
    all_events, all_event_id = mne.events_from_annotations(channel_raw, cfg.event_code_to_id)

    channel_epochs = mne.Epochs(channel_raw, all_events, event_id=all_event_id,
                                preload=True,
                                tmin=-0.5, tmax=2.6, baseline=(-0.5, 0),
                                reject_tmin=0,
                                reject={'seeg': 10},  # unit : V
                                )
    reg_out = regression(channel_epochs, 'action_class_specificity')