import project_config as cfg
import mne
import numpy as np


def nk_to_mne(save=True, return_raw=True):
    eeg_file = cfg.eeg_data_path / f'{subject_name}_ActionBase.EEG'
    raw = mne.io.read_raw_nihon(eeg_file)

    # raw.info['bads'].append("F'8")
    seeg_picks = mne.pick_channels_regexp(raw.ch_names, regexp='^[A-Z]\'(\d+)')
    stim_picks = mne.pick_channels_regexp(raw.ch_names, "DC")
    eeg_picks = mne.pick_channels_regexp(raw.ch_names, regexp='^[A-Z](\d+)')
    ref_picks = mne.pick_channels_regexp(raw.ch_names, regexp='^$')
    eog_picks = mne.pick_channels_regexp(raw.ch_names, "EOG")

    channel_type_dict = dict()
    for idx in seeg_picks:
        channel_type_dict[raw.ch_names[idx]] = 'seeg'
    for idx in stim_picks:
        channel_type_dict[raw.ch_names[idx]] = 'stim'
    for idx in eog_picks:
        channel_type_dict[raw.ch_names[idx]] = 'eog'
    for idx in eeg_picks:
        channel_type_dict[raw.ch_names[idx]] = 'seeg'

    raw.set_channel_types(channel_type_dict)

    events = mne.find_events(raw, min_duration=0.001, initial_event=False, consecutive=False)
    events = events[:-1]

    event_log_file = next(cfg.event_codes_path.glob(f'ActionBase_*{subject_name[1:]}.txt'))
    event_codes = np.loadtxt(event_log_file, dtype='i4')
    events[:, 2] = event_codes

    annot_from_events = mne.annotations_from_events(
        events=events, event_desc=cfg.event_id_to_code, sfreq=raw.info['sfreq'],
        orig_time=raw.info['meas_date'])
    raw.set_annotations(annot_from_events)

    # Only keep sEEG channels
    raw.pick_types(seeg=True)

    if save:
        save_file = cfg.steps_save_path / f'{subject_name}_STEP_0_annotated_raw.fif'
        raw.save(save_file, overwrite=True)
    if return_raw:
        return raw


subject_name = cfg.get_var('current_subject')
redo = cfg.get_var('redo')
