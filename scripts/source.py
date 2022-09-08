# %%
import mne
from seeg_action import project_config as cfg
import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

cfg.init_config('PriviteraM')

epochs = mne.read_epochs(cfg.epochs_file)
epochs.pick_types(seeg=True)

# %%
head_mri_t = mne.read_trans(cfg.subject_head_mri_t)

#%%
# if not cfg.oct_6_src_file.exists():
#     seeg_action = mne.setup_source_space(cfg.current_subject, spacing='oct6', add_dist=True,
#                                  n_jobs=4,
#                                  subjects_dir=cfg.patients_path)
#     seeg_action.save(cfg.oct_6_src_file, overwrite=True)

src = mne.read_source_spaces(cfg.oct_6_src_file)

# %%
# dc_evoked = epochs['DC'].average()
# sc_evoked = epochs['SC'].average()
# evoked_diff = mne.combine_evoked([dc_evoked, sc_evoked], weights=[1, -1])
#
# # %%
# gamma_power_t = evoked_diff.copy().filter(50, 150).apply_hilbert(
#     envelope=True)
#
# gamma_power_t.crop(tmin=-0.2, tmax=2.)
#
# # %%
#
# stc = mne.stc_near_sensors(
#     evoked_diff, head_mri_t, cfg.current_subject, subjects_dir=cfg.patients_path, seeg_action=seeg_action,
#     verbose='error')  # ignore missing electrode warnings
# stc = abs(stc)
#
# # %%
# stc_gamma = mne.stc_near_sensors(
#     gamma_power_t, head_mri_t, cfg.current_subject, subjects_dir=cfg.patients_path, seeg_action=seeg_action,
#     verbose='error')  # ignore missing electrode warnings
# # %%
# brain = stc_gamma.plot(surface='pial', hemi='lh', colormap='inferno', colorbar=True, clim='auto',
#                        views=['lat', 'med'], subjects_dir=cfg.patients_path, transparent=True,
#                        view_layout='horizontal', alpha=0.4, time_viewer=True)
# brain.add_sensors(evoked_diff.info, trans=head_mri_t)
#
# # %%
# stc_fs = mne.compute_source_morph(stc_gamma, cfg.current_subject, 'fsaverage', cfg.patients_path,
#                                   smooth=5, verbose='error').apply(stc_gamma)
#
# brain = stc_fs.plot(subjects_dir=cfg.patients_path,
#                     clim='auto', colormap='inferno',
#                     surface='flat', hemi='both',
#                     time_viewer=True,
#                     view_layout='horizontal')
#
# # %%
# # load the labels
# mt_lh = mne.read_label(cfg.subject_path / 'label' / 'lh.MT.label')
# mt_rh = mne.read_label(cfg.subject_path / 'labels' / 'rh.MT.label')
#
# # extract the time course for different labels from the stc
# stc_lh = stc.in_label(mt_lh)
# stc_rh = stc.in_label(mt_rh)
# stc_bh = stc.in_label(mt_lh + mt_rh)

# %%
evokeds = {cond: epochs[cond].average() for cond in epochs.event_id}

# %%
mn_spc_evoked = mne.combine_evoked([evokeds['MN/ST'], evokeds['IP/ST'], evokeds['SD/ST'],
                                    evokeds['MN/SC'], evokeds['IP/SC'], evokeds['SD/SC'],
                                    evokeds['MN/DC'], evokeds['IP/DC'], evokeds['SD/DC']
                                    ],
                                   weights=[1, -0.5, -0.5,
                                            -0.5, 0.25, 0.25,
                                            -0.5, 0.25, 0.25])

ip_spc_evoked = mne.combine_evoked([evokeds['MN/ST'], evokeds['IP/ST'], evokeds['SD/ST'],
                                    evokeds['MN/SC'], evokeds['IP/SC'], evokeds['SD/SC'],
                                    evokeds['MN/DC'], evokeds['IP/DC'], evokeds['SD/DC']
                                    ],
                                   weights=[-0.5, 1, -0.5,
                                            0.25, -0.5, 0.25,
                                            0.25, -0.5, 0.25])

sd_spc_evoked = mne.combine_evoked([evokeds['MN/ST'], evokeds['IP/ST'], evokeds['SD/ST'],
                                    evokeds['MN/SC'], evokeds['IP/SC'], evokeds['SD/SC'],
                                    evokeds['MN/DC'], evokeds['IP/DC'], evokeds['SD/DC']
                                    ],
                                   weights=[-0.5, -0.5,  1,
                                            0.25, 0.25, -0.5,
                                            0.25, 0.25, -0.5])

spc_evokeds = dict(mn_spc=mn_spc_evoked, ip_spc=ip_spc_evoked, sd_spc=sd_spc_evoked)

# %%
for ac, evk in spc_evokeds.items():
    stc = mne.stc_near_sensors(
        evk, head_mri_t, cfg.current_subject, subjects_dir=cfg.patients_path, src=src,
        verbose='error')  # ignore missing electrode warnings
    stc = abs(stc)

# %%
gamma = {}
stc_gamma = {}
for ac, evk in spc_evokeds.items():
    gamma[ac] = evk.copy().filter(50, 150).apply_hilbert(
        envelope=True).crop(tmin=-0.2, tmax=2.)

    stc_gamma[ac] = mne.stc_near_sensors(
        gamma[ac], head_mri_t, cfg.current_subject, subjects_dir=cfg.patients_path, src=src,
        verbose='error')  # ignore missing electrode warnings

# %%
def specific_gamma(ac):
    brain = stc_gamma[ac].plot(surface='pial', hemi='lh', colormap='inferno', colorbar=True, clim='auto',
                           views=['lat', 'med'], subjects_dir=cfg.patients_path, transparent=True,
                           title=f'{ac} gamma',
                           view_layout='horizontal', alpha=0.4, time_viewer=True)
    brain.add_sensors(spc_evokeds[ac].info, trans=head_mri_t)

# %%
specific_gamma('sd_spc')

# %%
def specific_gamma_flat(ac):
    stc_fs = mne.compute_source_morph(stc_gamma[ac], cfg.current_subject, 'fsaverage', cfg.patients_path,
                                      smooth=5, verbose='error').apply(stc_gamma[ac])

    brain = stc_fs.plot(subjects_dir=cfg.patients_path,
                        clim='auto', colormap='inferno',
                        surface='flat', hemi='both',
                        time_viewer=True,
                        view_layout='horizontal')


# %%
specific_gamma_flat('sd_spc')


# %%
# define frequencies of interest
freqs = np.geomspace(50, 152, num=10)
n_cycles = freqs / freqs[0]
high_gamma = {}
for cond, evk in evokeds.items():
    e = epochs[cond].load_data()
    e.pick_types(seeg=True)
    e.crop(tmax=1.5)
    epochs_tfr = mne.time_frequency.tfr_morlet(e, freqs, n_cycles=n_cycles, zero_mean=False,
                                               average=False, return_itc=False, n_jobs=-3, verbose='ERROR')
    epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, -0.2))
    epochs_tfr.crop(tmin=-0.3, tmax=1.2)
    high_gamma[cond] = epochs_tfr

# %%
data = [np.asarray(e.data) for _, e in high_gamma.items()]


# %%
def stat_fun(*args):
    return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                               effects=effects, return_pvals=False)[0]


effects = 'A'
n_conditions = len(epochs.event_id)
n_replications = epochs.events.shape[0] // n_conditions

factor_levels = [3, 3]  # number of levels in each factor

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = mne.stats.f_threshold_mway_rm(n_replications, factor_levels, effects,
                                         pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 256  # Save some time (the test won't be too sensitive ...)

# %%
F_obs, clusters, cluster_p_values, h0 = clu = mne.stats.permutation_cluster_test(
    data, stat_fun=stat_fun, threshold=f_thresh, tail=tail, buffer_size=None,
    n_jobs=-3, n_permutations=n_permutations)

# %%
# Select the clusters that are statistically significant at p < 0.05
good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
good_clusters = [clusters[idx] for idx in good_clusters_idx]

# %%
# print('Visualizing clusters.')
#
# # Now let's build a convenient representation of our results, where consecutive
# # cluster spatial maps are stacked in the time dimension of a SourceEstimate
# # object. This way by moving through the time dimension we will be able to see
# # subsequent cluster maps.
# stc_all_cluster_vis = mne.stats.summarize_clusters_stc(clu,
#                                                        subject=cfg.current_subject)
#
# # Let's actually plot the first "time point" in the SourceEstimate, which
# # shows all the clusters, weighted by duration.
#
# # blue blobs are for condition A < condition B, red for A > B
# brain = stc_all_cluster_vis.plot(
#     hemi='both', views='lateral', subjects_dir=cfg.patients_path,
#     size=(800, 800), smoothing_steps=5)
#


# %%
# trans = mne.channels.compute_native_head_t(montage)
#
# # %%
# fs_montage = mne.coreg.apply_trans(trans)
#
# aseg = 'aparc+aseg'  # parcellation/anatomical segmentation atlas
# labels, colors = mne.get_montage_volume_labels(
#     fs_montage, cfg.current_subject, subjects_dir=cfg.patients_path, aseg=aseg)

# %%
# mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=cfg.patients_path,
#                                         verbose=True)
# labels = mne.read_labels_from_annot(
#     'fsaverage', 'HCPMMP1', 'lh', subjects_dir=cfg.patients_path)
#
#
# # %%
# # TODO lets make a dict for each evoked according to hypothesis
# roi_names = ['L_AIP_ROI-lh', 'L_IPS1_ROI-lh']
# roi_labels = [label for label in labels if label.name in roi_names]

# %%
#

# epochs.crop(tmin=-0.2, tmax=1.)
# evk = {cond: epochs[cond].average() for cond in epochs.event_id}
# mne.viz.plot_compare_evokeds(evk, colors={'MN': 'red', 'IP': 'blue', 'SD': 'green'},
#                             linestyles={'ST': 'solid', 'SC': ':', 'DC': '--'})
# %%
# info = epochs.info
# electrodes = list(set(ch[0] for ch in epochs.ch_names))
#
#
# def my_callback(ax, ch_idx):
#     """
#     This block of code is executed once you click on one of the channel axes
#     in the plot. To work with the viz internals, this function should only take
#     two parameters, the axis and the channel or data index.
#     """
#     mne.viz.plot_compare_evokeds(evk,
#                                  picks=[ch_idx],
#                                  colors={'MN': 'red', 'IP': 'blue', 'SD': 'green'},
#                                  linestyles={'ST': 'solid', 'SC': ':', 'DC': '--'},
#                                  axes=ax,
#                                  show=False,
#                                  legend=True,
#                                  split_legend=True,
#                                  show_sensors=False)
#
#
# for e in electrodes:
#     fig = plt.figure(e)
#     e_picks = mne.pick_channels_regexp(epochs.ch_names, e)
#     print([info.ch_names[i] for i in e_picks])
#     layout = mne.channels.make_grid_layout(info, picks=e_picks, n_col=5)
#     for ax, idx in mne.viz.iter_topography(info,
#                                            layout=layout,
#                                            fig=fig,
#                                            fig_facecolor='white', axis_facecolor='white', axis_spinecolor='white',
#                                            on_pick=my_callback,
#                                            legend=False,
#                                            layout_scale=0.8):
#         mne.viz.plot_compare_evokeds(evk,
#                                      picks=[idx],
#                                      colors={'MN': 'red', 'IP': 'blue', 'SD': 'green'},
#                                      linestyles={'ST': 'solid', 'SC': ':', 'DC': '--'},
#                                      axes=ax,
#                                      show=False,
#                                      legend=False,
#                                      show_sensors=False)
#     fig.suptitle(f'Evoked responses in {e}')
#     handles, labels = plt.gca().get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right')
# plt.show()


# %%
# evoked = evk['MN/ST']


# %%
# import numpy as np
# vmin, vmid, vmax = np.percentile(evoked.data, [10, 25, 90])
# clim = dict(kind='value', lims=[vmin, vmid, vmax])

# %%
# brain = stc.plot(surface='pial', hemi='both', colormap='mne', colorbar=True,
#                  clim='auto', views=['lat', 'med'], subjects_dir=cfg.patients_path, transparent=True,
#                  view_layout='horizontal', alpha=0.4, smoothing_steps='nearest', time_viewer=True)
# brain.add_sensors(evoked.info, trans=head_mri_t)
#
# # %%
# stc_fs = mne.compute_source_morph(stc, cfg.current_subject, 'fsaverage', cfg.patients_path,
#                                   smooth=5, verbose='error').apply(stc)
#
# brain = stc_fs.plot(subjects_dir=cfg.patients_path,
#                     clim='auto', colormap='mne',
#                     surface='flat', hemi='both',
#                     smoothing_steps='nearest', time_viewer=True,
#                     view_layout='horizontal')
#
# # to help orient us, let's add a parcellation (red=auditory, green=motor,
# # blue=visual)
# brain.add_annotation('HCPMMP1_combined', borders=2)
#
# # %%
# anat_labels = mne.read_labels_from_annot(cfg.current_subject, parc='aparc',
#                                          subjects_dir=cfg.patients_path)

# %%

# aparc_label_name = 'parsopercularis-lh'
# # Make an STC in the time interval of interest and take the mean
# stc_mean = stc.copy().crop(0, 1.).mean()
#
# # use the stc_mean to generate a functional label
# # region growing is halted at 60% of the peak value within the
# # anatomical label / ROI specified by aparc_label_name
# label = mne.read_labels_from_annot(cfg.current_subject, parc='aparc',
#                                    subjects_dir=cfg.patients_path,
#                                    regexp=aparc_label_name)[0]
# stc_mean_label = stc_mean.in_label(label)
# data = np.abs(stc_mean_label.data)
# stc_mean_label.data[data < 0.6 * np.max(data)] = 0.
#
# # 8.5% of original source space vertices were omitted during forward
# # calculation, suppress the warning here with verbose='error'
# func_labels, _ = mne.stc_to_label(stc_mean_label, seeg_action=seeg_action, smooth=True,
#                                   subjects_dir=cfg.patients_path, connected=True,
#                                   verbose='error')
