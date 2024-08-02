from pathlib import Path
import argparse
import numpy as np
import sys


# Argument Parser
# ref: https://realpython.com/command-line-interfaces-python-argparse/
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Imports raw EEG from Nihon Kohden files,"
                    "Imports annotations from log txt file"
                    "Epochs continuous EEG according to annotations")
    parser.add_argument('-s', '--subject_name',
                        type=str,
                        required=True)
    parser.add_argument('-r', '--redo',
                        type=bool,
                        default=False)
    return parser


# this is a pointer to the module object instance itself.
ProjectConfig = sys.modules[__name__]
ProjectConfig.__slots__ = (
    'project_path', 'current_subject', 'data_path',
    'patients_path', 'subject_path', 'derivatives_path',
    'exec_log_path', 'raw_data_path', 'figure_save_path',
    'raw_fif_save_file', 'filtered_raw_file', 'epochs_file',
    'epochs_action_file', 'bad_channels_file', 'ica_file',
    'montage_file', 'oct_6_src_file', 'vol_src_file',
    'subject_head_mri_t', 'covariance_mat_file',
    'event_log_file', 'event_id_to_description',
    'event_description_to_id', 'action_classes',
    'filtered_raw_file', 'epochs_file', 'raw_fif_save_file',
    'presented_stimuli', 'ica_file', 'f1_score_file')

p = Path().absolute()
ProjectConfig.project_path = p.parent

# Local
if p.owner() == 'senaer':
    ProjectConfig.project_path = Path('/Users/senaer/Codes/CCNLab/sEEG-action-classes/')
    ProjectConfig.data_path = ProjectConfig.project_path / 'data'

# Server
elif p.owner() == 'ser':
    ProjectConfig.project_path = Path('/auto/k2/ser/sEEG-action-classes')
    ProjectConfig.data_path = Path('/auto/data2/ser/StereoEEG_ActionBase')

ProjectConfig.patients_path = ProjectConfig.data_path / 'subjects'
ProjectConfig.exec_log_path = ProjectConfig.project_path / 'log'
ProjectConfig.figure_save_path = ProjectConfig.project_path / 'results' / 'figures'
ProjectConfig.action_classes = ['OBJ', 'PER', 'BOD']
ProjectConfig.presented_stimuli = ['ST', 'CS', 'CD']

def init_current_subject(current_subject_name):
    ProjectConfig.current_subject = current_subject_name

    ProjectConfig.subject_path = ProjectConfig.patients_path / ProjectConfig.current_subject
    ProjectConfig.derivatives_path = ProjectConfig.subject_path / 'derivatives'

    ProjectConfig.raw_data_path = ProjectConfig.subject_path / 'ses-01' / 'ieeg'

    _paths = [
        ProjectConfig.patients_path,
        ProjectConfig.exec_log_path,
        ProjectConfig.figure_save_path,
        ProjectConfig.derivatives_path,
        ProjectConfig.raw_data_path,
        Path(ProjectConfig.subject_path / 'src'),
        Path(ProjectConfig.subject_path / 'bem'),
    ]
    for path in _paths:
        path.mkdir(parents=True, exist_ok=True)

    ProjectConfig.T1_file = ProjectConfig.subject_path / 'mri' / 'T1.mgz'
    ProjectConfig.seeg_locations_file = ProjectConfig.raw_data_path / f'{ProjectConfig.current_subject}_seeg_locations.json'
    ProjectConfig.fiducials_file = ProjectConfig.subject_path / 'bem' / f'{ProjectConfig.current_subject}-fiducials.fif'

    ProjectConfig.oct_6_src_file = ProjectConfig.subject_path / 'src' / f'{ProjectConfig.current_subject}-oct-6-src.fif.gz'
    if not ProjectConfig.oct_6_src_file.exists():
        from mne import setup_source_space
        src = setup_source_space(ProjectConfig.current_subject, spacing='oct6', add_dist=True,
                                     n_jobs=-4, subjects_dir=ProjectConfig.patients_path)
        src.save(ProjectConfig.oct_6_src_file, overwrite=True)


    ProjectConfig.vol_src_file = ProjectConfig.subject_path / 'src' / f'{ProjectConfig.current_subject}-vol-src.fif.gz'
    ProjectConfig.vol_source_morph_file = ProjectConfig.subject_path / 'src' / f'{ProjectConfig.current_subject}-vol-morph.h5'
    ProjectConfig.surface_source_morph_file = ProjectConfig.subject_path / 'src' / f'{ProjectConfig.current_subject}-surface-morph.h5'

    ProjectConfig.raw_fif_save_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-raw.fif.gz'
    ProjectConfig.filtered_raw_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-filtered-raw.fif.gz'
    ProjectConfig.epochs_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-epo.fif.gz'
    ProjectConfig.clean_epochs_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-clean-epo.fif.gz'
    ProjectConfig.bipolar_ref_epochs_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-bpr-epo.fif.gz'
    ProjectConfig.bad_channels_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-bad-channels.txt'
    ProjectConfig.ica_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-ica.fif.gz'
    ProjectConfig.montage_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-montage.fif'
    ProjectConfig.subject_head_mri_t = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-subject-mri-head-trans.fif'
    ProjectConfig.covariance_mat_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-cov.fif'
    ProjectConfig.f1_score_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-3way-f1-score-ave.fif'
    ProjectConfig.permutation_scores_file = ProjectConfig.derivatives_path / f'{ProjectConfig.current_subject}-permutation-scores.pkl'


    ProjectConfig.event_log_file = next(
        ProjectConfig.raw_data_path.glob(f'ActionBase_*{ProjectConfig.current_subject[1:-1]}*_detailed.txt'))

    # TODO may fix detailed_event_id byte order
    trial_no, detailed_event_id, detailed_description, simple_event_id = \
        np.genfromtxt(ProjectConfig.event_log_file, delimiter='\t',
                      dtype=None, encoding=None,
                      converters={2: lambda s: s.replace('_', '/')[:14]},
                      unpack=True)

    ProjectConfig.event_id_to_description = dict(zip(detailed_event_id, detailed_description))
    ProjectConfig.event_description_to_id = dict(zip(detailed_description, detailed_event_id))

    # ProjectConfig.action_classes = set(map(lambda k: k.split('/')[0], detailed_description))
    # ProjectConfig.presented_stimuli = set(map(lambda k: k.split('/')[1], detailed_description))


ProjectConfig.init_current_subject = init_current_subject
