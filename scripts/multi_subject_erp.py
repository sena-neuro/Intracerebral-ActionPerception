# %%
import mne
from seeg_action.project_config import ProjectConfig as cfg
from scipy.stats import trim_mean

# %%
src_template = mne.read_source_spaces(cfg.patients_path / 'fsaverage' / 'bem' / 'fsaverage-vol-5-src.fif')
preprocessed_subjects = ['PriviteraM', 'PetriceanuC', 'ZanoniM']

# %%
stcs_dct = {}
for subject in preprocessed_subjects:
    cfg.init_current_subject(subject)

    head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
    epochs = mne.read_epochs(cfg.epochs_file)
    epochs.pick_types(seeg=True)

    evokeds_dict = {}

    for ac in cfg.action_classes:
        for st in cfg.presented_stimuli:
            cond = '/'.join((ac, st))
            evokeds_dict[cond] = evk = epochs[cond].average(method=lambda x: trim_mean(x, 0.1, axis=0))

            stc = mne.stc_near_sensors(
                evk, head_mri_t, 'fsaverage',
                mode='nearest',
                subjects_dir=cfg.patients_path,
                src=src_template,
                distance=.005,
                project=False,
                verbose='error')
            stcs_dct[cond].append(stc)








