# %%
import mne
from seeg_action import project_config as cfg
import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

cfg.init_config('PriviteraM')

head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
src = mne.read_source_spaces(cfg.oct_6_src_file)
src_fs = mne.read_source_spaces(cfg.patients_path / 'fsaverage' / 'bem' / 'fsaverage-ico-5-src.fif')

epochs = mne.read_epochs(cfg.epochs_file)
epochs.pick_types(seeg=True)

# %%
montage = epochs.get_montage()
montage.apply_trans(mne.transforms.invert_transform(head_mri_t))

labels, colors = mne.get_montage_volume_labels(
    montage, cfg.current_subject, subjects_dir=cfg.patients_path, aseg='aseg')

wm_channels = [c for c, l in labels.items() if len(l) > 0 and "White-Matter" in l[0]]
epochs.drop_channels(wm_channels)

# %%
evokeds = {cond: epochs[cond].average() for cond in epochs.event_id}

action_classes = set(map(lambda k: k.split('/')[0], cfg.event_code_to_id.keys()))

# presented_stimulus = set(map(lambda k: k.split('/')[-1], cfg.event_code_to_id.keys()))
# for _ac in action_classes.difference(ac)

action_minus_control_diff = {}
action_class_specificity_diff = {}
for ac in action_classes:
    action_minus_control_diff[ac] = mne.combine_evoked([evokeds[ac + '/ST'], evokeds[ac + '/SC'], evokeds[ac + '/DC']],
                                                       weights=[1, -0.5, -0.5])

    action_class_specificity_diff[ac] = mne.combine_evoked(
        [evokeds[ac + '/ST'], evokeds[ac + '/SC'], evokeds[ac + '/DC'], ],
        weights=[1, -0.5, -0.5])

action_class_specificity_diff['MN'] = mne.combine_evoked(
    [evokeds['MN/ST'], evokeds['IP/ST'], evokeds['SD/ST'],
     evokeds['MN/SC'], evokeds['IP/SC'], evokeds['SD/SC'],
     evokeds['MN/DC'], evokeds['IP/DC'], evokeds['SD/DC']],
    weights=[1, -0.5, -0.5,
             -0.5, 0.25, 0.25,
             -0.5, 0.25, 0.25])

action_class_specificity_diff['IP'] = mne.combine_evoked(
    [evokeds['MN/ST'], evokeds['IP/ST'], evokeds['SD/ST'],
     evokeds['MN/SC'], evokeds['IP/SC'], evokeds['SD/SC'],
     evokeds['MN/DC'], evokeds['IP/DC'], evokeds['SD/DC']
     ],
    weights=[-0.5, 1, -0.5,
             0.25, -0.5, 0.25,
             0.25, -0.5, 0.25])

action_class_specificity_diff['SD'] = mne.combine_evoked(
    [evokeds['MN/ST'], evokeds['IP/ST'], evokeds['SD/ST'],
     evokeds['MN/SC'], evokeds['IP/SC'], evokeds['SD/SC'],
     evokeds['MN/DC'], evokeds['IP/DC'], evokeds['SD/DC']
     ],
    weights=[-0.5, -0.5, 1,
             0.25, 0.25, -0.5,
             0.25, 0.25, -0.5])


# %%
def calculate_stcs(evks):
    subject_stc = {}
    fsaverage_stc = {}
    for cond, evk in evks.items():
        subject_stc[cond] = mne.stc_near_sensors(
            evk, head_mri_t, cfg.current_subject, subjects_dir=cfg.patients_path, src=src,
            verbose='error')  # ignore missing electrode warnings

        fsaverage_stc[cond] = mne.compute_source_morph(
            subject_stc[cond], cfg.current_subject, 'fsaverage', cfg.patients_path,
            smooth=5, verbose='error').apply(subject_stc[cond])
    return subject_stc, fsaverage_stc


# %%
_, fsaverage_stc = calculate_stcs(evokeds)
_, fsaverage_stc_diff = calculate_stcs(action_minus_control_diff)
_, fsaverage_stc_spc = calculate_stcs(action_class_specificity_diff)

# %%
brain = fsaverage_stc['MN/ST'].plot(
    subjects_dir=cfg.patients_path,
    clim='auto', colormap='inferno',
    surface='flat', hemi='split',
    time_viewer=True,
    view_layout='vertical')

# %%

brain = fsaverage_stc_diff['MN'].plot(
    subjects_dir=cfg.patients_path,
    clim='auto', colormap='inferno',
    surface='flat', hemi='split',
    time_viewer=True,
    view_layout='vertical')

# %%
brain = fsaverage_stc_spc['MN'].plot(
    subjects_dir=cfg.patients_path,
    clim='auto', colormap='inferno',
    surface='flat', hemi='split',
    time_viewer=True,
    view_layout='vertical')

# %%
roi_patterns = '|'.join('(%s)' % roi for roi in
                        [r'AIP', r'OP\d.*', r'PF\w+', r'PG/w+', r'FST', r'IPS',
                         r'V\d+\w*', r'STS\w*', r'STG.*', 'MS?T'])

pattern = r"(([LR]_)(" + roi_patterns + ")(_ROI-[lr]h))"


# load the anatomical ROI for comparison
anat_labels = mne.read_labels_from_annot('fsaverage', hemi='both', parc='HCPMMP1',
                                         subjects_dir=cfg.patients_path,
                                         regexp=pattern)

# %%
label_tcs_map = {}
label_stc_map = {}
for label in anat_labels:
    l_name = label.name
    label_tcs_map[l_name] = {}
    label_stc_map[l_name] = {}
    for cond, stc in fsaverage_stc.items():
        label_tcs_map[l_name][cond] = stc.extract_label_time_course(label, src_fs,
                                                                    allow_empty=False, mode='auto')
        label_stc_map[l_name][cond] = stc.in_label(label)


# %%
def plot_time_course_in_label(label_name):
    label_tcs = label_tcs_map[label_name]
    label_stc = label_stc_map[label_name]
    fig, ax = plt.subplots(1)

    style_map = dict(
        c={'MN': 'red', 'SD': 'blue', 'IP': 'green'},
        ls={'ST': 'solid', 'SC': 'dashed', 'DC': 'dotted'},
        alpha={'ST': 1, 'SC': 0.5, 'DC': 0.5},
        lw={'ST': 2, 'SC': 1, 'DC': 1}
    )

    for cond, tc in label_tcs.items():
        ax.plot(label_stc[cond].times, np.squeeze(tc), label=cond,
                c=style_map['c'][cond[:2]],
                ls=style_map['ls'][cond[3:]],
                alpha=style_map['alpha'][cond[3:]],
                lw=style_map['lw'][cond[3:]],
                )
    ax.legend(loc='upper right')
    ax.set(xlabel='Time (s)', ylabel='Source amplitude',
           title='Activations in Label %r' % label_name)
    mne.viz.tight_layout()

# %%
plot_time_course_in_label('L_STSvp_ROI-lh')

# %%
plot_time_course_in_label('L_PFop_ROI-lh')

# %%
plot_time_course_in_label('L_MST_ROI-lh')

# %%
plot_time_course_in_label('L_MT_ROI-lh')

# %%
plot_time_course_in_label('L_V4_ROI-lh')

# %%
# for label in anat_labels:
#    plot_time_course_in_label(label)

# %%
label_tcs_map_spc = {}
label_stc_map_spc = {}
for label in anat_labels:
    l_name = label.name
    label_tcs_map_spc[l_name] = {}
    label_stc_map_spc[l_name] = {}
    for cond, stc in fsaverage_stc_spc.items():
        label_tcs_map_spc[l_name][cond] = stc.extract_label_time_course(label, src_fs,
                                                                        allow_empty=False, mode='auto')
        label_stc_map_spc[l_name][cond] = stc.in_label(label)


# %%
def plot_time_course_in_label_spc(label_name):
    label_tcs = label_tcs_map_spc[label_name]
    label_stc = label_stc_map_spc[label_name]
    fig, ax = plt.subplots(1)

    style_map = dict(
        c={'MN': 'red', 'SD': 'blue', 'IP': 'green'},
    )

    for cond, tc in label_tcs.items():
        ax.plot(label_stc[cond].times, np.squeeze(tc), label=cond,
                c=style_map['c'][cond[:2]],

                )
    ax.legend(loc='upper right')
    ax.set(xlabel='Time (s)', ylabel='Source amplitude',
           title='Activations in Label %r' % label_name)
    mne.viz.tight_layout()

# %%
plot_time_course_in_label_spc('L_STSvp_ROI-lh')

# %%
plot_time_course_in_label_spc('L_OP4_ROI-lh')

# %%
plot_time_course_in_label_spc('L_V1_ROI-lh')
