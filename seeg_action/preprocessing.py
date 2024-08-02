import mne
import numpy as np
from seeg_action.project_config import ProjectConfig as cfg

# import sys
# this is a pointer to the module object instance itself.
# this = sys.modules[__name__]


def export_bipolar_reference_epochs():
    import re
    epochs = mne.read_epochs(cfg.epochs_file)
    epochs.pick_types(seeg=True)

    pattern = r'^([A-Z]\'?)\d+'
    shafts = list(set(re.search(pattern, ch).groups()[0] for ch in epochs.ch_names))

    epochs_bpr = epochs.copy()
    for shaft in shafts:
        channels = [ch for ch in epochs.ch_names if re.search(pattern, ch).groups()[0] == shaft]
        epochs_bpr = mne.set_bipolar_reference(epochs_bpr,
                                               anode=channels[:-1],
                                               cathode=channels[1:])

    epochs_bpr.save(cfg.bipolar_ref_epochs_file, overwrite=True)

def get_gm_channels(info):
    head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
    montage = info.get_montage()
    if head_mri_t.to_str == 'head':
        head_mri_t = mne.transforms.invert_transform(head_mri_t)
    montage.apply_trans(head_mri_t)

    labels, colors = mne.get_montage_volume_labels(
        montage, cfg.current_subject, subjects_dir=cfg.patients_path, aseg='aseg')

    gm_channels = [channel for channel, labels in labels.items() if any("Cerebral-Cortex" in l for l in labels)]
    return gm_channels


def get_seeg_montage(info):
    import nibabel as nib
    import json

    lps_to_ras = np.eye(4)
    lps_to_ras[0, 0] = -1.
    lps_to_ras[1, 1] = -1.

    t1 = nib.load(cfg.T1_file)

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

    # Some channels are coded as X'1.0 instead of X'1 in the json file,
    # so we take the substring preceeding the dot.
    with open(cfg.seeg_locations_file) as file:
        data = json.load(file)
    channel_pos_dict = {chan['label'].split('.')[0]: np.array(chan['position']) for chan in
                        data['markups'][0]['controlPoints']}
    channel_pos_dict_new = {
        k: transform(v)
            for k, v in channel_pos_dict.items()
                if k in info['ch_names']
    }


    montage = mne.channels.make_dig_montage(ch_pos=channel_pos_dict_new,
                                            coord_frame='mri')
    fids, coord = mne.io.read_fiducials(cfg.fiducials_file)
    montage.dig += fids

    montage.save(cfg.montage_file, overwrite=True)

    mri_head_t = mne.channels.compute_native_head_t(montage)
    mne.write_trans(cfg.subject_head_mri_t, mri_head_t, overwrite=True)

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

        #layout = mne.channels.make_grid_layout(raw.info, n_col=15)

        # plot diagnostics
        #ica_solution.plot_properties(raw, picks=eog_indices, topomap_args=dict(pos=layout.pos))

        # plot ICs applied to raw data, with EOG matches highlighted
        ica_solution.plot_sources(raw, show_scrollbars=False)

        eog_evoked = mne.preprocessing.create_eog_epochs(raw).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))

        # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        ica_solution.plot_sources(eog_evoked)


def export_raw_fif():
    import re
    raw_file = next(cfg.raw_data_path.glob('*.[EEG EDF]*'))
    if raw_file.suffix == '.EDF':
        # exclude = "E$|MILO|KG|DEL\d"
        raw = mne.io.read_raw_edf(raw_file, infer_types=False, stim_channel='DC09')
    elif raw_file.suffix == '.EEG':
        raw = mne.io.read_raw_nihon(raw_file)

    # --------------------------------------------------------------------------
    # Channels
    # --------------------------------------------------------------------------

    # Here assumption is that the first one is an sEEG channel
    duplicates = mne.pick_channels_regexp(raw.ch_names, '\w\d+-0')
    if len(duplicates) != 0:
        raw.rename_channels({raw.ch_names[idx]: raw.ch_names[idx].split('-')[0]
                             for idx in duplicates})

    ch_patterns = {
        'misc': r'($[A-Z]\d|E-|MILO|TIB|E)(\d?)',
        'eog': r'EOG',
        'ecg': r'EKG|KG|DEL',
        'ref': r'$',
        'stim': r'DC',
        'seeg': r'^[A-Z][\']?(\d+)',

    }


    channel_type_dict = dict()
    for ch_type, pattern in ch_patterns.items():
        channel_type_dict.update(
            dict.fromkeys(list(filter(re.compile(pattern).match, raw.ch_names)),
                          ch_type))
    raw.set_channel_types(channel_type_dict)

    # set montage
    raw.set_montage(get_seeg_montage(raw.info), on_missing='warn')
    non_seeg_chs = [k for k, v in raw.get_montage().get_positions()['ch_pos'].items() if np.isnan(v[0])]

    raw.set_channel_types(
        {
            ch: 'eeg' for ch in non_seeg_chs if raw.get_channel_types(ch)[0] == 'seeg'
        }
    )

    # Currently, this does nothing...
    # Load bad channels
    if cfg.bad_channels_file.exists():
        raw.load_bad_channels(cfg.bad_channels_file)
    # Create empty file
    else:
        cfg.bad_channels_file.touch()

    # Sort channels
    order = raw.copy().pick_types(seeg=True).info['ch_names'].copy()

    # Sort based on hemisphere, electrode shaft, and number
    def _order_key(x):
        if "'" in x:
            hemi = 10000
            alpha, num = x.split("'")
        else:
            hemi = -10000
            alpha, num = x[0], x[1:]
        return hemi + ord(alpha) * 100 + int(num)

    order.sort(key=_order_key)
    non_seeg_order = sorted(list(set(raw.info['ch_names']).difference(order)))
    order.extend(non_seeg_order)
    raw = raw.reorder_channels(order)

    # --------------------------------------------------------------------------
    # Events
    # --------------------------------------------------------------------------
    def discretize(arr):
        from sklearn.preprocessing import KBinsDiscretizer
        return KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans') \
            .fit_transform(arr.reshape(-1, 1)).ravel()

    stim_raw = raw.copy().pick_types(stim=True).load_data() # or pick DC09
    stim_raw.apply_function(discretize, picks=['DC09'])
    events = mne.find_stim_steps(stim_raw, merge=-10)
    events = mne.pick_events(events, include=[1])[:576]

    event_log_file = next(cfg.raw_data_path.glob(f'ActionBase_*{cfg.current_subject[1:-1]}*_detailed.txt'))

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

    # --------------------------------------------------------------------------
    # Info
    # --------------------------------------------------------------------------
    def camel_case_split(str):
        return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str)

    subject_info = {'his_id': cfg.current_subject,
                    'last_name': camel_case_split(cfg.current_subject)[0],
                    'first_name': camel_case_split(cfg.current_subject)[-1]
                    }
    new_info = mne.Info()
    new_info['subject_info'] = subject_info
    raw.info.update(new_info)

    raw.save(cfg.raw_fif_save_file, overwrite=True)


def export_filtered_raw_fif():
    raw = mne.io.read_raw_fif(cfg.raw_fif_save_file)
    raw_filtered = bandpass_filter(raw)
    raw_filtered = _filter_power_line_noise(raw_filtered)
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
def bandpass_filter(raw, l_freq=1.5, h_freq=300.):
    raw.load_data()
    raw.filter(l_freq=l_freq, h_freq=None, method='iir', iir_params=dict(order=6, ftype='butter'))
    raw.filter(l_freq=None, h_freq=h_freq, method='iir', iir_params=dict(order=6, ftype='butter'))
    return raw


def _filter_power_line_noise(raw, freqs=None, notch_method='spectrum'):
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

    return raw_notch

