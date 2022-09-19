import mne
import numpy as np
import sys
from seeg_action import project_config as cfg
import re

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]


def get_wm_channels(info):
    head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
    montage = info.get_montage()
    montage.apply_trans(mne.transforms.invert_transform(head_mri_t))

    labels, colors = mne.get_montage_volume_labels(
        montage, cfg.current_subject, subjects_dir=cfg.patients_path, aseg='aseg')

    wm_channels = [c for c, l in labels.items() if len(l) > 0 and "White-Matter" in l[0]]
    return wm_channels


def get_seeg_montage(info):
    import nibabel as nib
    import json

    lps_to_ras = np.eye(4)
    lps_to_ras[0, 0] = -1.
    lps_to_ras[1, 1] = -1.

    t1 = nib.load(cfg.subject_path / 'mri' / 'T1.mgz')

    # An affine array that tells you the position of the image array data in a reference space.
    # affine : voxel to ras_coords_scanner_mm
    # inv_affine : ras_coords_scanner_mm to voxel
    inv_affine = np.linalg.inv(t1.affine)

    # Translates voxels to surface mri coord
    vox2surface = t1.header.get_vox2ras_tkr()

    def transform(x):
        pos = np.array(x)
        pos = mne.transforms.apply_trans(lps_to_ras, pos)
        pos = mne.transforms.apply_trans(inv_affine, pos)
        pos = mne.transforms.apply_trans(vox2surface, pos)
        pos /= 1000.
        return pos

    # Some channels are coded as X'1.0 instead of X'1 in the json file
    # so we take the substring preceeding the dot.
    with open(cfg.subject_path / 'electrodes' / 'left_electrodes.json') as file:
        data_left = json.load(file)
    channel_pos_dict = {chan['label'].split('.')[0] : np.array(chan['position']) for chan in
                        data_left['markups'][0]['controlPoints']}
    with open(cfg.subject_path / 'electrodes' / 'right_electrodes.json') as file:
        data_right = json.load(file)
    channel_pos_dict.update(
        {chan['label'].split('.')[0]: np.array(chan['position']) for chan in data_right['markups'][0]['controlPoints']}
    )
    channel_pos_dict_new = {k: transform(v) for k, v in channel_pos_dict.items() if k in info['ch_names']}

    montage = mne.channels.make_dig_montage(ch_pos=channel_pos_dict_new,
                                            coord_frame='mri')

    fids, coord = mne.io.read_fiducials(cfg.subject_path / 'bem' / f'{cfg.current_subject}-fiducials.fif')
    montage.dig += fids

    montage.save(cfg.montage_file, overwrite=True)

    return montage


def get_covariance_matrix():
    try:
        noise_cov = mne.read_cov(cfg.covariance_mat_file)
    except FileNotFoundError:
        try:
            epochs = mne.read_epochs(cfg.epochs_file)
        except FileNotFoundError:
            print('Exporting epochs...')
            export_epochs()
            epochs = mne.read_epochs(cfg.epochs_file)

        # Noise covariance matrix for baseline
        noise_cov = mne.compute_covariance(epochs, tmax=0.,
                                           method='auto',  # Regularization
                                           rank=None)
        mne.write_cov(cfg.covariance_mat_file, noise_cov)
    return noise_cov


def export_ica_solution():
    cov = get_covariance_matrix()
    raw = mne.io.read_raw_fif(cfg.filtered_raw_file).load_data()
    ica = mne.preprocessing.ICA(max_iter='auto', noise_cov=cov, method='picard')
    ica.fit(raw)
    ica.save(cfg.ica_file, overwrite=True)

    ica.plot_sources(raw, block=True)
    ica.plot_overlay(raw, exclude=ica.exclude)
    ica.save(cfg.ica_file, overwrite=True)


def find_eog_components(raw, ica_solution):
    ica_solution.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica_solution.find_bads_eog(raw)
    if len(eog_indices) > 0:
        ica_solution.exclude = eog_indices

        # barplot of ICA component "EOG match" scores
        ica_solution.plot_scores(eog_scores)

        # layout = mne.channels.make_grid_layout(raw.info, n_col=15)

        # plot diagnostics
        # ica_solution.plot_properties(raw, picks=eog_indices, topomap_args=dict(pos=layout.pos))

        # plot ICs applied to raw data, with EOG matches highlighted
        ica_solution.plot_sources(raw, show_scrollbars=False)

        eog_evoked = mne.preprocessing.create_eog_epochs(raw).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))

        # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        ica_solution.plot_sources(eog_evoked)


def export_raw_fif():
    raw_file = next(cfg.raw_data_path.glob('*.[EEG EDF]*'))
    if raw_file.suffix == '.EDF':
        raw = mne.io.read_raw_edf(raw_file, infer_types=True, stim_channel='DC09', exclude="E$|MILO|KG|DEL\d")
    elif raw_file.suffix == '.EEG':
        raw = mne.io.read_raw_nihon(raw_file)

    seeg_picks = mne.pick_channels_regexp(raw.ch_names, regexp='^[A-Z][\']?(\d+)')
    stim_picks = mne.pick_channels_regexp(raw.ch_names, "DC")
    eog_picks = mne.pick_channels_regexp(raw.ch_names, "EOG")

    channel_type_dict = dict()
    for idx in seeg_picks:
        channel_type_dict[raw.ch_names[idx]] = 'seeg'
    for idx in stim_picks:
        channel_type_dict[raw.ch_names[idx]] = 'stim'
    for idx in eog_picks:
        channel_type_dict[raw.ch_names[idx]] = 'eog'
    raw.set_channel_types(channel_type_dict)

    def discretize(arr):
        from sklearn.preprocessing import KBinsDiscretizer
        return KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans') \
            .fit_transform(arr.reshape(-1, 1)).ravel()

    stim_raw = raw.copy().pick('DC09').load_data()
    stim_raw.apply_function(discretize, picks=['DC09'])
    events = mne.find_stim_steps(stim_raw, merge=-10)

    event_log_file = next(cfg.raw_data_path.glob(f'ActionBase_*{cfg.current_subject[1:-1]}_detailed.txt'))

    trial_no, detailed_event_id, detailed_description, simple_event_id = \
        np.genfromtxt(event_log_file, delimiter='\t',
                      dtype=None, encoding=None,
                      converters={2: lambda s: s.replace('_', '/')[:14]},
                      unpack=True)
    events[:, 2] = detailed_event_id

    annot = mne.annotations_from_events(
        events=events, sfreq=stim_raw.info['sfreq'],
        event_desc=dict(zip(detailed_event_id, detailed_description)),
        orig_time=stim_raw.info['meas_date'])
    raw.set_annotations(annot)

    def camel_case_split(str):
        return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str)

    subject_info = {'his_id': cfg.current_subject,
                    'last_name': camel_case_split(cfg.current_subject)[0],
                    'first_name': camel_case_split(cfg.current_subject)[-1]
                    }
    new_info = mne.Info()
    new_info['subject_info'] = subject_info
    raw.info.update(new_info)

    raw.pick_types(seeg=True, eog=True)

    # set montage
    raw.set_montage(get_seeg_montage(raw.info), on_missing='warn')

    # Load bad channels
    if cfg.bad_channels_file.exists():
        raw.load_bad_channels(cfg.bad_channels_file)
    # Create empty file
    else:
        cfg.bad_channels_file.touch()

    raw.save(cfg.raw_fif_save_file, overwrite=True)


def export_filtered_raw_fif():
    raw = mne.io.read_raw_fif(cfg.raw_fif_save_file)
    raw_filtered = this._bandpass_filter(raw)
    raw_filtered = this._filter_power_line_noise(raw_filtered)
    raw_filtered.save(cfg.filtered_raw_file, overwrite=True)


def export_epochs():
    raw = mne.io.read_raw_fif(cfg.filtered_raw_file).load_data()

    all_events, event_id = mne.events_from_annotations(raw, cfg.event_description_to_id)
    epochs = mne.Epochs(raw, all_events, event_id=event_id,
                        preload=True,
                        tmin=-0.5, tmax=2.6,
                        baseline=(None, 0),
                        detrend=1,
                        verbose=True
                        )

    epochs.save(cfg.epochs_file, overwrite=True)


def export_action_minus_control_epochs():
    epochs = mne.read_epochs(cfg.epochs_file)
    epochs_action = epochs['ST']
    for cond in epochs_action.event_id:
        action_class, _, action_examplar, object_size, actor = cond.split('/')
        evk_cs = epochs['/'.join([action_class, 'CS', object_size, actor])].average()
        evk_cd = epochs['/'.join([action_class, 'CD', object_size, actor])].average()
        evk_control = mne.combine_evoked([evk_cs, evk_cd], [0.5, 0.5])
        epochs_action[cond].subtract_evoked(evk_control)

    epochs_action.save(cfg.epochs_action_file, overwrite=True)


# Auxiliary functions for filtering
def _bandpass_filter(raw, l_freq=1.5, h_freq=300.):
    raw.load_data()
    raw.filter(l_freq=l_freq, h_freq=None, method='iir', iir_params=dict(order=6, ftype='butter'))
    raw.filter(l_freq=None, h_freq=h_freq, method='iir', iir_params=dict(order=6, ftype='butter'))
    return raw


def _filter_power_line_noise(raw, freqs=None, notch_method='spectrum', visualize=False):
    if freqs is None:
        freqs = np.arange(50, 251, 50)

    if notch_method == 'fir':
        raw_notch = raw.notch_filter(freqs=freqs, picks='seeg', trans_bandwidth=0.04)
    elif notch_method == 'iir':
        raw_notch = raw.notch_filter(
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
