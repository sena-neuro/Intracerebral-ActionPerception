# %%
import matplotlib.pyplot as plt
import mne
from seeg_action.project_config import ProjectConfig as cfg

# %%
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1',
                                    subjects_dir=cfg.patients_path)

source_space_kind = 'volume' # or surface
if source_space_kind == 'volume':
    # Volumetric Source Space
    src_template = mne.read_source_spaces(cfg.patients_path / 'fsaverage' / 'bem' / 'fsaverage-vol-5-src.fif')
else:
    # Surface Source Space
    src_template = mne.read_source_spaces(cfg.patients_path / 'fsaverage' / 'bem' / 'fsaverage-ico-5-src.fif')

conditions = ['OBJ/ST', 'PER/ST', 'BOD/ST']
stcs_dct = dict.fromkeys(conditions, [])

# 'PriviteraM'
decoded_subjects = ['PetriceanuC', 'ZanoniM']

# %%
subject = 'PetriceanuC'
cfg.init_current_subject(subject)

head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
evokeds = mne.read_evokeds(cfg.f1_score_file)

# Check head_mri_t is correct for 'fsaverage'
stc_dct = dict.fromkeys(conditions)
for evk in evokeds:
    # evk.apply_function(lambda x: x * 100.)

    cond = evk.comment
    stc = mne.stc_near_sensors(
        evk, head_mri_t, 'fsaverage',
        mode='nearest',
        subjects_dir=cfg.patients_path,
        src=src_template,
        distance=.005,
        project=False,
        verbose='error')
    stc_dct[cond] = stc

# %%
mne.viz.set_browser_backend('matplotlib')

# %%
stc_dct['OBJ/ST'].plot(src_template,
                       clim=dict(kind='value', lims=[0.33, 0.4, 0.5]))

# %%
stc_dct['OBJ/ST'].plot(src_template,
                       mode='glass_brain',
                       clim=dict(kind='value', lims=[0.33, 0.4, 0.5]))


# %%
for subject in decoded_subjects:
    cfg.init_current_subject(subject)

    head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
    evokeds = mne.read_evokeds(cfg.f1_score_file)

    # Check head_mri_t is correct for 'fsaverage'
    stc_dct = dict.fromkeys(conditions)
    for evk in evokeds:
        # evk.apply_function(lambda x: x * 100.)

        cond = evk.comment
        stc = mne.stc_near_sensors(
            evk, head_mri_t, 'fsaverage',
            mode='nearest',
            subjects_dir=cfg.patients_path,
            src=src_template,
            distance=.005,
            project=False,
            verbose='error')

# %%
stc_dct['BOD/ST'].plot(src_template)

# %%
merged_stc_dct = dict(zip(stcs_dct, map(sum, stcs_dct.values())))

# %%
merged_stc_dct['PER/ST'].plot(
    src=src_template,
    clim=dict(kind='value', lims=(0.25, 0.33, 0.5)),
    colormap='viridis',
    transparent=True)

# %%
# mne.viz.set_browser_backend('matplotlib')
# stc.plot(src=src_template)
# %%
# draw_label_borders = True
# plot_template_3d = True

# %%
# activated_labels_dct = dict.fromkeys(conditions, [])
# if draw_label_borders:
#     for cond in conditions:
#         tcs = stc_template_dct[cond].extract_label_time_course(
#             labels,
#             src_template,
#             mode='mean',
#             allow_empty='ignore')
#         for label, tc in zip(labels, tcs):
#             if any(tc > 0.5):
#                 activated_labels_dct[cond].append(label)
#
# # %%
# cond = 'BOD/ST'
# if plot_template_3d:
#     brain_fs = mne.viz.plot_source_estimates(
#         stc=stc_template_dct[cond],
#         subjects_dir=cfg.patients_path,
#         title=cond, alpha=0.7, initial_time=0,
#         clim='auto', colormap='inferno',
#         surface='inflated', hemi='both',
#         time_viewer=True)
#     if draw_label_borders:
#         for label in activated_labels_dct[cond]:
#             brain_fs.add_label(label, alpha=0.7)

# %%
# # T
#     if source_space_kind == 'volume':
#         # Volumetric Source Space
#         src_subject = mne.setup_volume_source_space(cfg.current_subject, subjects_dir=cfg.patients_path, volume_label=rois)
#         if cfg.vol_source_morph_file.exists():
#             source_morph = mne.read_source_morph(cfg.vol_source_morph_file)
#         else:
#             source_morph = mne.compute_source_morph(src=src_subject,
#                                                     subject_from=cfg.current_subject, subject_to='fsaverage',
#                                                     subjects_dir=cfg.patients_path, src_to=src_template,
#                                                     warn=False,
#                                                     smooth=5,  # 'nearest'
#                                                     verbose='error')
#             source_morph.compute_vol_morph_mat()
#             source_morph.save(cfg.vol_source_morph_file)
#
#     else:
#         # Surface Source Space
#         src_subject = mne.read_source_spaces(cfg.oct_6_src_file)
#
#         if cfg.surface_source_morph_file.exists():
#             source_morph = mne.read_source_morph(cfg.surface_source_morph_file)
#         else:
#             source_morph = mne.compute_source_morph(src=src_subject,
#                                                     subject_from=cfg.current_subject, subject_to='fsaverage',
#                                                     subjects_dir=cfg.patients_path, src_to=src_template,
#                                                     warn=False,
#                                                     smooth=5,  # 'nearest'
#                                                     verbose='error')
#             source_morph.save(cfg.surface_source_morph_file)
