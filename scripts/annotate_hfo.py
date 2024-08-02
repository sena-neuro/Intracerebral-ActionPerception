import mne_hfo
import mne
from seeg_action import project_config as cfg

# MNE-HFO and pyqtgraph is not compatible
# Therefore, run MNE-HFO in a seperate console
# Save the annotated result in a fif file
# Then, read the annotated result file in another console
# Now, you can plot the HFO annotations with pyqtgraph
def detect(raw, detector_type, filter_band, win_size, threshold, overlap=None, override=True):

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

    kwargs.pop("overlap")
    kwargs.pop("sfreq")
    kwargs['detector'] = detector_type
    annot_filename = "_".join(f"{key.replace('_', '')}={value}" for key, value in kwargs.items()) + ".txt"

    # Replace comma-space in win_size value
    # since annotations values in .txt are read
    # by splitting comma (in mne.read_annotations())
    # e.g., (80, 250) -> (80-250)
    annot_filename = annot_filename.replace(', ', '-').replace('(', '').replace(')', '')

    # Check filename
    print(annot_filename)

    annot_path = cfg.steps_save_path / annot_filename

    # If overriding old annotations is not desired
    # and HFO annotations with same parameters is already created
    # No need to detect HFO with the same parameters
    if not override and annot_path.exists():
        # Does nothing
        return

    detector.fit(raw)
    df = detector.hfo_df

    # Create and save annotations
    annot = mne.Annotations(onset=df.onset,
                            duration=df.duration,
                            description=['HFO'],
                            orig_time=None,
                            ch_names=[[ch] for ch in df.channels])
    annot.save(annot_path, overwrite=True)


def hfo(raw):
    # %%
    ripple_band = (80, 250)
    fast_ripple_band = (250, 499)

    detect(raw, detector_type='LineLength', filter_band=ripple_band, win_size=10, threshold=5)


if __name__ == '__main__':
    # Take arguments and initialize configurations
    parser = cfg.init_argparse()
    args = parser.parse_args()

    # raw_fif_file = cfg.steps_save_path / f'{args.subject_name}_STEP_0_annotated_raw.fif'
    raw = mne.io.read_raw_fif(cfg.steps_save_path / f'{args.subject_name}_notch_filtered_50_500_raw.fif')
    ch_subset = ["T'10", "S'10", "G'15", "B'14", "H'15", "I'11", "I'12", "Q'10", "M'10", "R'10"]
    raw = raw.pick(ch_subset)
    hfo(raw)
