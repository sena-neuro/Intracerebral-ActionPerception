import mne
import numpy as np
import project_config as cfg


def get_hfo_annotations(raw, detector='LineLength',
                        threshold=10,
                        filter_band=(80, 250),
                        win_size=10,
                        visualize=True):
    hfo_params = {
        'detector': detector,
        'threshold': str(threshold),
        'filterband': str(filter_band).replace(', ', '-'),
        'winsize': str(win_size),
    }

    annot_filename = "_".join(f"{key}={value}" for key, value in hfo_params.items()) + '.txt'
    annot_file = cfg.steps_save_path / annot_filename

    return _examine_hfo(raw, annot_file, hfo_params, visualize)


def examine_all_hfo_detectors(raw):
    visualize = True
    for annot_file in cfg.steps_save_path.rglob(".txt"):
        hfo_params = dict()
        for param in annot_file.stem.split('_'):
            key, value = param.split('=')
            hfo_params[key] = value
            _examine_hfo(raw, annot_file, hfo_params, visualize)


def _examine_hfo(raw, annot_file, hfo_params, visualize):
    hfo_annot = mne.read_annotations(annot_file)

    # Adding BAD means drop the entire epoch:
    # hfo_annot.rename(dict(zip(hfo_annot.description,
    #                  map(lambda x: x.replace('HFO at', 'BAD'), hfo_annot.description))))

    if visualize:
        lead_letter = 'O'

        lead_picks = mne.pick_channels_regexp(raw.ch_names, regexp=f'^{lead_letter}')
        hfo_raw_lead = raw.copy().set_annotations(hfo_annot).pick(lead_picks)

        # hfo_params.pop('sfreq')
        title = " ".join(f"{key}={value}" for key, value in hfo_params.items())
        hfo_raw_lead.plot(scalings='auto', block=True, precompute=True, title=title)
    return hfo_annot


def extract_gamma_power(epochs):
    freqs = np.arange(50, 160, 10)
    n_cycles = freqs * .1
    gamma_power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, use_fft=True,
                                                n_jobs=4)
