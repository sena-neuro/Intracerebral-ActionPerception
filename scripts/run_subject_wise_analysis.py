import mne
from seeg_action import project_config as cfg
from seeg_action import preprocessing as pp
from seeg_action import mass_univariate_analysis as mua

if __name__ == '__main__':

    # Take arguments and initialize configurations
    parser = cfg.init_argparse()
    args = parser.parse_args()
    subject_name = args.subject_name

    # Read filtered raw  EEG
    filtered_raw_file = cfg.steps_save_path / f'{subject_name}_filtered_raw.fif'

    if filtered_raw_file.exists():
        raw = mne.io.read_raw_fif(filtered_raw_file)
    else:
        raw_fif_file = cfg.steps_save_path / f'{subject_name}_raw.fif'
        raw = mne.io.read_raw_fif(raw_fif_file)
        raw = raw.load_data()
        iir_params = dict(order=6, ftype='butter')
        raw = raw.filter(method='iir', iir_params=iir_params,
                         l_freq=1.5, h_freq=300)

        raw = pp.filter_power_line_noise(raw)
        filtered_raw_file = cfg.steps_save_path / f'{subject_name}_filtered_raw.fif'
        raw.save(filtered_raw_file, overwrite=True)

    # for channel in raw.ch_names:
    channel = "O'9"
    channel_raw = raw.copy().pick(channel)

    all_events, event_id = mne.events_from_annotations(channel_raw, cfg.event_code_to_id)

    epochs = mne.Epochs(channel_raw, all_events, event_id=event_id,
                        preload=True,
                        tmin=-0.5, tmax=2.6,
                        baseline=(None, 0),
                        detrend=1,
                        reject={'seeg': 3},  # unit : V
                        flat={'seeg': 1e-2},
                        verbose=False
                        )

    epochs.equalize_event_counts()