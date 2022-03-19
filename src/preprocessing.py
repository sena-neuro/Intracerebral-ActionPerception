import mne
import numpy as np
import project_config as cfg


def epoch(raw):
    all_events, all_event_id = mne.events_from_annotations(raw)
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


subject_name = cfg.get_var('current_subject')
redo = cfg.get_var('redo')