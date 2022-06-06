import mne
from seeg_action import project_config as cfg
from seeg_action import preprocessing as pp
from seeg_action import mass_univariate_analysis as mua
import pandas as pd


if __name__ == '__main__':
    # Take arguments and initialize configurations
    parser = cfg.init_argparse()
    args = parser.parse_args()
    cfg.init_config(args.subject_name)

    epochs_file = cfg.steps_save_path / f'{cfg.current_subject}-rej-epo.fif'
    if epochs_file.exists():
        epochs = mne.read_epochs(epochs_file)
        mua.regression(epochs)

    else:
        # Read filtered raw  EEG
        filtered_raw_file = cfg.steps_save_path / f'{cfg.current_subject}_filtered_raw.fif'

        if filtered_raw_file.exists():
            raw = mne.io.read_raw_fif(filtered_raw_file)
        else:
            raw_fif_file = cfg.steps_save_path / f'{cfg.current_subject}_raw.fif'
            raw = mne.io.read_raw_fif(raw_fif_file)
            raw = raw.load_data()
            iir_params = dict(order=6, ftype='butter')
            raw = raw.filter(method='iir', iir_params=iir_params,
                             l_freq=1.5, h_freq=300)

            raw = pp.filter_power_line_noise(raw)
            filtered_raw_file = cfg.steps_save_path / f'{cfg.current_subject}_filtered_raw.fif'
            raw.save(filtered_raw_file, overwrite=True)

        annot_file = cfg.steps_save_path / 'threshold=5_filterband=80-250_winsize=10_detector=LineLength.txt'
        hfo_annot = mne.read_annotations(annot_file)
        df = hfo_annot.to_data_frame()

        initial_ts = df.onset[0].floor('D')
        df.onset = df.onset - initial_ts
        df[['duration']] = df[['duration']].apply(pd.to_timedelta, unit='s')
        df[['ch_names']] = df[['ch_names']].apply(lambda x: list(x)[0], axis=1, result_type="expand")
        df['offset'] = df.onset + df.duration
        df.sort_values("onset", inplace=True)
        ## This line compares if onset of next row is greater than FINISH of current
        ## row ("shift" shifts down offset by one row). The value of expression before
        ## cumsum will be True if interval breaks (i.e. cannot be merged), so
        ## cumsum will increment group value when interval breaks (cumsum treats True=1, False=0)
        tol = pd.Timedelta('1 sec')
        df["group"] = (df["onset"] > df["offset"].shift() + tol).cumsum()
        ## this returns min value of "onset" column from a group and max value from "offset"
        result = df.groupby(["group"])\
            .agg({"onset": "min", "offset": "max", "ch_names": set})\
            .assign(description='HFO', duration= lambda x : x.offset - x.onset)

        result['ch_names'] = result['ch_names'].apply(list)
        result[['onset', 'duration', 'offset']] = result[['onset', 'duration', 'offset']].apply(lambda x: x.dt.total_seconds())
        result.loc[result["ch_names"].str.len() > 2, 'description'] = 'BAD_HFO'
        # result.drop(result[(result.duration < 0.012) & (result.ch_names.map(len) < 2)].index, axis='rows', inplace=True)
        hfo_annotations = mne.Annotations(onset=result.onset,
                                          duration=result.duration,
                                          description=result.description,
                                          ch_names=result.ch_names.values)
        raw.set_annotations(raw.annotations + hfo_annotations)
        raw.info['bads'].extend(["F'8"])

        all_events, event_id = mne.events_from_annotations(raw, cfg.event_code_to_id)

        epochs = mne.Epochs(raw, all_events, event_id=event_id,
                            preload=True,
                            tmin=-0.5, tmax=2.6,
                            baseline=(None, 0),
                            detrend=1,
                            reject={'seeg': 3},  # unit : V
                            flat={'seeg': 1e-2},
                            verbose=False
                            )
        epochs.equalize_event_counts()
        # %%

        epochs.save(epochs_file, overwrite=True)