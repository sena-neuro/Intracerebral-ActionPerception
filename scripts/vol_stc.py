# %%
import matplotlib.pyplot as plt

import mne
from scipy.stats import trim_mean, zmap, trimboth
import numpy as np
from seeg_action.project_config import ProjectConfig as cfg
from seeg_action import preprocessing as pp

import re

# %%
subject = 'FedericoR'
cfg.init_current_subject(subject)

# %%
freq_band_map = {
#    'Theta': (4, 7),
    'Mu': (7.5, 12.5),
#    'Alpha': (8, 12),
    'SMR': (12.5, 15.5),
#    'Beta': (12, 30),
#    'Low-Gamma': (32, 100),
#    'High-Gamma': (80, 150)
    'Gamma': (50, 150)
}
conditions = ['/'.join((ac, st)) for st in cfg.presented_stimuli for ac in cfg.action_classes]

# %%
# (re)load the data to save memory
orig_raw = mne.io.read_raw_fif(cfg.filtered_raw_file)
# raw.pick_types(seeg=True, eog=True)
orig_raw.load_data()
orig_raw = orig_raw.apply_function(lambda x: zmap(x, trimboth(x, 0.1)))

# %%
power_map = {freq: {cond: None for cond in conditions} for freq in freq_band_map}

for band, (fmin, fmax) in freq_band_map.items():
    raw = orig_raw.copy()
    raw = pp.bandpass_filter(raw, l_freq=fmin, h_freq=fmax)

    # epoch
    all_events, event_id = mne.events_from_annotations(raw, cfg.event_description_to_id)
    epochs = mne.Epochs(raw, all_events, event_id=event_id,
                        preload=True,
                        tmin=-1., tmax=3.,
                        baseline=(-0.5, -0.1),
                        verbose=True
                        )
    epochs.apply_baseline()


    # get analytic signal (envelope)
    epochs.apply_hilbert(envelope=True)
    epochs.crop(tmin=-0.5, tmax=2.6)

    for cond in conditions:
        power_map[band][cond] =  epochs[cond].average(method=lambda x : trim_mean(x, 0.1, axis=0))

# %%
evk = power_map['Mu']['OBJ/ST']

# %%
# colormap = plt.get_cmap('tab20')
# shafts = sorted(list(set([ch[0] for ch in evk.ch_names])))
# letter_to_index = {letter: i for i, letter in enumerate(shafts)}
# sensor_colors = [colormap(letter_to_index[ch[0]]) for ch in evk.ch_names]
#
# # %%
# mne.viz.plot_alignment(
#     evk.info, head_mri_t,
#     surfaces=dict(brain=0.01, outer_skull=0.01, head=0.01),
#     sensor_colors=sensor_colors,
#     subject=cfg.current_subject, seeg=True, src=None, mri_fiducials=False,
# )

# %%
head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
mri_head_t = head_mri_t.copy()
head_mri_t = mne.transforms.invert_transform(head_mri_t)
mri_mni_t = mne.read_talxfm(cfg.current_subject, cfg.patients_path)



# %%
surf_file = cfg.subject_path / 'bem' / 'inner_skull.surf'
mri_file = cfg.subject_path / 'mri' / 'aseg.mgz'
vol_src = mne.setup_volume_source_space(
    cfg.current_subject,
    pos=5.,
    subjects_dir=cfg.patients_path,
    mri=mri_file,
    volume_label='Left-Cerebral-Cortex',
    surface=surf_file,
    mindist=5.,
    single_volume=True,
    add_interpolator=True,
    verbose=True)

# %%
# Montage in head frame
montage = evk.get_montage()

# Montage in MRI (surface RAS) frame
montage.apply_trans(head_mri_t)

evk.set_montage(montage)

# %%
trans = mne.channels.compute_native_head_t(montage)

# %%
evk = power_map['Mu']['PER/ST']

# %%
vol_stc = mne.stc_near_sensors(
    evk, trans=trans,
    subject=cfg.current_subject,
    mode='weighted', # weighted
    subjects_dir=cfg.patients_path, src=vol_src,
    project=True,
    distance=0.01,
)
vol_stc.apply_baseline(baseline=(-0.5, -0.1))


# %%
clim = dict(kind='value', pos_lims=[0., 0.05, 0.3])# np.percentile(abs(evk.data), [10, 50, 75]))


# %%
clim = dict(kind='value', lims=[0., 0.05, 0.3])# np.percentile(abs(evk.data), [10, 50, 75]))

# %%
vol_stc.plot(src=vol_src, subject=cfg.current_subject, transparent='auto', clim=clim)

# %%
brain = vol_stc.plot_3d(
    src=vol_src, subjects_dir=cfg.patients_path,
    hemi='lh', initial_time=0.,
    view_layout='horizontal', views=['axial', 'coronal', 'sagittal'],
    size=(800, 300), show_traces=0.4,
    add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=8)))


# %%
brain.add_sensors(evk.info, trans=trans, seeg=True)

# %%
# # %%
# freq_band_map = {
# #    'Theta': (4, 7),
#     'Mu': (7.5, 12.5),
# #    'Alpha': (8, 12),
#     'SMR': (12.5, 15.5),
# #    'Beta': (12, 30),
# #    'Low-Gamma': (32, 100),
# #    'High-Gamma': (80, 150)
#     'Gamma': (50, 150)
# }
# conditions = ['/'.join((ac, st)) for st in cfg.presented_stimuli for ac in cfg.action_classes]
#
# # %%
# stcs = {freq: {cond: None for cond in conditions} for freq in freq_band_map}
#
# for band in freq_band_map:
#     for cond in conditions:
#         fname = cfg.derivatives_path / f'{cfg.current_subject}_{band}_{"_".join(cond.split("/"))}-lh.stc'
#         stcs[band][cond] = mne.read_source_estimate(fname, subject=cfg.current_subject)
#
# # %%
# stc = stcs['Mu']['BOD/ST']
# stc.apply_baseline(baseline=(-0.5, -0.1))
# brain = stc.plot(subject=cfg.current_subject, hemi='lh', clim='auto', size=600)
#
# # %%


# %%
# # Montage in head frame
# montage = evk.get_montage()
#
# # Montage in MRI (surface RAS) frame
# montage.apply_trans(head_mri_t)
#
# # Montage in MNI Talairach frame frame
# montage.apply_trans(mri_mni_t)
#
# # for fsaverage, "mri" and "mni_tal" are equivalent and, since
# # we want to plot in fsaverage "mri" space, we need use an identity
# # transform to equate these coordinate frames
# montage.apply_trans(
#     mne.transforms.Transform(fro='mni_tal', to='mri', trans=np.eye(4)))
#
# evk.set_montage(montage)
#
# # compute the transform to head for plotting
# trans = mne.channels.compute_native_head_t(montage)
# # note that this is the same as:
# # ``mne.transforms.invert_transform(
# #      mne.transforms.combine_transforms(head_mri_t, mri_mni_t))``
#
# src_template = mne.read_source_spaces(cfg.patients_path / 'fsaverage' / 'bem' / 'fsaverage-vol-5-src.fif')
#
# vol_stc = mne.stc_near_sensors(
#     evk, trans=trans,
#     subject='fsaverage',
#     mode='weighted',
#     subjects_dir=cfg.patients_path, src=src_template,
#     project=False,
#     distance=0.01,
# )
# vol_stc.apply_baseline(baseline=(-0.5, -0.1))
#
# # %%
# vol_stc.plot(src_template, 'fsaverage', )
#
# # %%
# brain = vol_stc.plot_3d(
#     src=src_template, subjects_dir=cfg.patients_path,
#     view_layout='horizontal', views=['axial', 'coronal', 'sagittal'],
#     size=(800, 300), show_traces=0.4,
#     add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=8)))
#
#