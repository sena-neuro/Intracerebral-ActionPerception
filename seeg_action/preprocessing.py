import mne
import numpy as np
import project_config as cfg


def epoch(raw):
    all_events, all_event_id = mne.events_from_annotations(raw, cfg.event_code_to_id)
    picks = mne.pick_types(raw.info, seeg=True, eeg=False, eog=False, stim=False)
    epochs = mne.Epochs(raw, all_events, event_id=all_event_id, picks=picks,
                        tmin=-0.5, tmax=2.6, baseline=(-0.5, 0),
                        reject_tmin=0,
                        reject={'seeg': 10}  # unit : V
                        # flat={'seeg': 1e-4}  # unit : V
                        )

    save_file = cfg.steps_save_path / f'{subject_name}_STEP_1_epoched_data-epo.fif'
    epochs.save(save_file, overwrite=True)
    return epochs


def ica(raw):
    # %%
    # ICA
    raw.pick_types(seeg=True, stim=True, eog=True)
    raw.load_data()
    filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
    ica = mne.preprocessing.ICA(n_components=30, max_iter='auto')
    ica.fit(filt_raw)
    raw.load_data()
    ica.plot_sources(raw)


def bandpass_filter(raw):
    iir_params = dict(order=6, ftype='butter')
    filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'],
                                             method='iir', iir_params=iir_params,
                                             l_freq=1.5, h_freq=300)


def filter_power_line_noise(raw, freqs=None, notch_method='spectrum', visualize=False):
    if freqs is None:
        freqs = np.arange(50, 251, 50)

    if notch_method == 'fir':
        raw_notch = raw.copy().notch_filter(freqs=freqs, picks='seeg', trans_bandwidth=0.04)
    elif notch_method == 'iir':
        raw_notch = raw.copy().notch_filter(
            freqs=freqs, picks='seeg', method='iir',
            iir_params=dict(order=6, ftype='butter'))
    else:
        raw.load_data()
        raw_notch = raw.notch_filter(
            freqs=freqs, picks='seeg', method='spectrum_fit', filter_length='10s')
    if visualize:
        _visualize_power_line_filtering(raw, raw_notch, notch_method)
    return raw_notch


def _add_arrows(axes):
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (50, 100, 150):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4):(idx + 5)].max()
            ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True)


def _visualize_power_line_filtering(raw, raw_notch, notch_method):
    for title, data in zip(['Un', f'Notch ({notch_method})'], [raw, raw_notch]):
        fig = data.plot_psd(fmax=155, average=True)
        fig.subplots_adjust(top=0.85)
        fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
        _add_arrows(fig.axes[:2])


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
