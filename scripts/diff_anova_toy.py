# %%
import mne
from seeg_action import project_config as cfg
from seeg_action import preprocessing as pp
import numpy as np

cfg.init_config('PriviteraM')

# %%
flag = 'evoked_stc'

# %%
epochs = mne.read_epochs(cfg.epochs_file)
epochs.pick_types(seeg=True)
epochs.drop_channels(pp.get_wm_channels(epochs.info))

# %%
if flag == 'tfr_plot':
    import matplotlib.pyplot as plt

    freqs = np.geomspace(5, 152, num=50)
    n_cycles = freqs / freqs[0]
    tfr_tmin, tfr_tmax = -0.3, 2.3

    sig_channels = ["V'9", "Z'12"]
    epochs.pick(sig_channels)

    epochs_tfr = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles=n_cycles,
                                               average=False, return_itc=False, n_jobs=-3)
    epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0))
    epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)

    power = {}
    for action_class in cfg.action_classes:
        power[action_class] = epochs_tfr[action_class].average()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    cols = cfg.action_classes
    rows = sig_channels

    for col, ax in zip(cols, zip(*axes)):
        power[col].plot(axes=list(ax), vmin=-3, vmax=3, combine=None, show=False)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large')

    plt.show()


# %%
if flag == 'evoked_stc':
    # evokeds = {cond: epochs[cond].average() for cond in cfg.action_classes}
    evokeds = {cond: epochs[cond].average() for cond in set(key[:6] for key in epochs.event_id.keys())}

    src = mne.read_source_spaces(cfg.oct_6_src_file)
    src_fs = mne.read_source_spaces(cfg.patients_path / 'fsaverage' / 'bem' / 'fsaverage-ico-5-src.fif')

    def calculate_stcs(evks):
        subject_stc = {}
        fsaverage_stc = {}
        for cond, evk in evks.items():
            subject_stc[cond] = mne.stc_near_sensors(
                evk, cfg.subject_head_mri_t, cfg.current_subject, subjects_dir=cfg.patients_path, src=src,
                verbose='error')  # ignore missing electrode warnings

            fsaverage_stc[cond] = mne.compute_source_morph(
                subject_stc[cond], cfg.current_subject, 'fsaverage', cfg.patients_path,
                smooth=5, verbose='error').apply(subject_stc[cond])
        return subject_stc, fsaverage_stc

    subject_stc, fsaverage_stc = calculate_stcs(evokeds)

    ac = 'PER/ST'
    #sig_channels = ["V'9", "Z'12"]
    #sig_info = epochs.copy().pick(sig_channels).info

    brain = subject_stc[ac].plot(
        surface='pial', hemi='both',
        views=['lat', 'med'], subjects_dir=cfg.patients_path, transparent=True,
        view_layout='horizontal', alpha=0.4, time_viewer=True)
    brain.add_sensors(epochs.info, cfg.subject_head_mri_t)

    flat_brain = fsaverage_stc[ac].plot(
        surface='flat', hemi='split', subjects_dir=cfg.patients_path, transparent=True,
        view_layout='vertical', alpha=0.4, time_viewer=True)
