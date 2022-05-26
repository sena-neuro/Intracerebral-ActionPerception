import mne
import scipy.io
from seeg_action import project_config as cfg
from mne.datasets import fetch_fsaverage
import numpy as np

subject_name = 'BerberiM'
electrode_report_path = cfg.patients_path / subject_name / 'electrod_report'

channel_pos_dict = dict()
for lead_report_mat_file in electrode_report_path.glob('*_report.mat'):
    lead_report_mat = scipy.io.loadmat(lead_report_mat_file)
    lead_electrode_report = lead_report_mat['Report'][0]
    column_names = [col_name for col_name, _ in lead_electrode_report.dtype.descr]
    channel_coord = lead_electrode_report[0]['central_coordinate_recording_leed_native']
    # 'central_coordinate_recording_leed_subject_fslr't

    channel_name = lead_electrode_report[0]['electode_name'][0]

    for i, pos in enumerate(channel_coord):
        ch_name = f"{channel_name}'{i+1}"
        channel_pos_dict[ch_name] = pos / 1000        # mm? to m

montage = mne.channels.make_dig_montage(channel_pos_dict, coord_frame='head')

# Read filtered raw  EEG
filtered_raw_file = cfg.steps_save_path / f'{subject_name}_filtered_raw.fif'

raw = mne.io.read_raw_fif(filtered_raw_file)

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
    channel_type_dict[raw.ch_names[idx]] = 'eeg'

raw.set_channel_types(channel_type_dict)

raw.pick_types(seeg=True)
raw.set_montage(montage)


subjects_dir = cfg.patients_path

# use mne-python's fsaverage data
fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed
# %%

montage = raw.get_montage()

# first we need a head to mri transform since the data is stored in "head"
# coordinates, let's load the mri to head transform and invert it
head_mri_t = mne.coreg.estimate_head_mri_t('BerberiM', subjects_dir)
# apply the transform to our montage
montage.apply_trans(head_mri_t)

# now let's load our Talairach transform and apply it
mri_mni_t = mne.read_talxfm('BerberiM', subjects_dir)
montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)

montage.add_mni_fiducials(subjects_dir=subjects_dir, verbose=True)

# for fsaverage, "mri" and "mni_tal" are equivalent and, since
# we want to plot in fsaverage "mri" space, we need use an identity
# transform to equate these coordinate frames
montage.apply_trans(
    mne.transforms.Transform(fro='mni_tal', to='mri', trans=np.eye(4)))

# Electrode locations in fsaverage mri space
raw.set_montage(montage)

# %%
aseg = 'aparc+aseg'  # parcellation/anatomical segmentation atlas
labels, colors = mne.get_montage_volume_labels(
    montage, 'fsaverage', subjects_dir=subjects_dir, aseg=aseg)

electrodes = set([''.join([lttr for lttr in ch_name
                           if not lttr.isdigit() and lttr != ' '])
                  for ch_name in montage.ch_names])
print(f'Electrodes in the dataset: {electrodes}')

# %%
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)
