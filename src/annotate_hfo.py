import mne_hfo
import mne
from src import project_config as cfg


# MNE-HFO and pyqtgraph is not compatible
# Therefore, run MNE-HFO in a seperate console
# Save the annotated result in a fif file
# Then, read the annotated result file in another console
# Now, you can plot the HFO annotations with pyqtgraph

def detect(orig_raw, detector_type, filter_band, win_size, threshold, overlap=None):
    raw = orig_raw.copy()

    if overlap is None:
        overlap = 1.0 / float(win_size)

    kwargs = {
        'threshold': threshold,
        'sfreq': raw.info['sfreq'],
        'filter_band': filter_band,
        'win_size': win_size,
        'overlap': overlap
    }

    if detector_type == 'LineLength':
        detector = mne_hfo.LineLengthDetector(**kwargs)
    elif detector_type == 'RMS':
        detector = mne_hfo.RMSDetector(**kwargs)
    else:
        raise RuntimeError(f'Cannot recognize parameter detector_type {detector_type}.'
                           f'Recognized values for detector_type are "RMS" and "LineLength".')

    description = f"detector={detector_type}_" + "_".join(
        f"{key.replace('_', '')}={value}" for key, value in kwargs.items())
    description = description.replace(', ', '-').replace('sfreq=1000.0_', '')
    print(description)

    annot_filename = description + ".txt"
    annot_path = cfg.steps_save_path / annot_filename
    # if annot_path.exists():
    #    return

    detector.fit(raw)

    df = detector.hfo_df

    # Create and save annotations
    annot = mne.Annotations(onset=df.onset,
                            duration=df.duration,
                            description=[f'HFO at {ch}' for ch in df.channels],
                            orig_time=None,
                            ch_names=[[ch] for ch in df.channels])

    annot.save(annot_path, overwrite=True)
    return annot


def hfo(orig_raw, subject_name):
    # %%
    raw = orig_raw.copy()
    raw.pick_types(seeg=True)

    ripple_band = (80, 250)
    fast_ripple_band = (250, 499)

    for detector in ["RMS", "LineLength"]:
        for filter_band in [ripple_band, fast_ripple_band]:
            for window_size in [1000, 500]: # [10, 20, 25, 50]:
                for threshold in [20, 15]: #[7, 10, 15]:
                    detect(raw,
                           detector_type=detector,
                           filter_band=filter_band,
                           win_size=window_size,
                           threshold=threshold)


if __name__ == '__main__':
    # Take arguments and initialize configurations
    parser = cfg.init_argparse()
    args = parser.parse_args()

    # raw_fif_file = cfg.steps_save_path / f'{args.subject_name}_STEP_0_annotated_raw.fif'
    notch_raw = mne.io.read_raw_fif(cfg.steps_save_path / f'{args.subject_name}_notch_filtered_50_500_raw.fif')
    hfo(notch_raw, args.subject_name)
