# %%
import mne
from seeg_action.project_config import ProjectConfig as cfg

# %%
cfg.init_current_subject('PetriceanuC')
head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
src = mne.read_source_spaces(cfg.oct_6_src_file)

# %%
evokeds = mne.read_evokeds(cfg.f1_score_file)
stcs = {}
for evk in evokeds:
    stcs[evk.comment] = mne.stc_near_sensors(
        evk, head_mri_t, cfg.current_subject,
        mode='weighted',
        subjects_dir=cfg.patients_path,
        src=src,
        distance=.005,
        project=False,
        verbose='error')

# %%
action_class = 'OBJ/ST'
plot_subject_3d = False
plot_label_time_trace = False
plot_template_flat = False
plot_template_3d = False
draw_label_borders = False
# %%
if plot_subject_3d:
    brain_obj = mne.viz.plot_source_estimates(
        stcs[action_class],
        hemi='both',
        surface='pial',
        subject=cfg.current_subject,
        subjects_dir=cfg.patients_path,
        alpha=.5
    )
    brain_obj.add_sensors(evokeds[0].info, head_mri_t)

# %%
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1',
                                   subjects_dir=cfg.patients_path)
src_fs = mne.read_source_spaces(cfg.patients_path / 'fsaverage' / 'bem' / 'fsaverage-ico-5-src.fif')

# %%
stc = stcs[action_class]
stc_fs = mne.compute_source_morph(stc, cfg.current_subject, 'fsaverage', cfg.patients_path,
                                  smooth=5, verbose='error').apply(stc)
tcs = stc_fs.extract_label_time_course(labels, src_fs, mode='max', allow_empty='ignore')

# %%
if draw_label_borders:
    activated_labels = []
    for label, tc in zip(labels, tcs):
        if any(tc > 0.5):
            activated_labels.append(label)

# %%
if plot_template_flat:
    brain_fs = mne.viz.plot_source_estimates(
        stc=stc_fs,
        subjects_dir=cfg.patients_path,
        clim='auto', colormap='inferno',
        surface='flat', hemi='split',
        time_viewer=True,
        view_layout='vertical')
    if draw_label_borders:
        for label in activated_labels:
            brain_fs.add_label(label, alpha=0.7)
# %%
if plot_template_3d:
    brain_fs = mne.viz.plot_source_estimates(
        stc=stc_fs,
        subjects_dir=cfg.patients_path,
        clim='auto', colormap='inferno',
        surface='pial', hemi='both',
        time_viewer=True)
    if draw_label_borders:
        for label in activated_labels:
            brain_fs.add_label(label, alpha=0.7)

# %%
if plot_label_time_trace:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for label, tc in zip(labels, tcs):
        if any(tc > 0.5):
            print(label.name)
            ax.plot(stc_fs.times, tc, label=label.name)
    fig.legend()
    fig.show()