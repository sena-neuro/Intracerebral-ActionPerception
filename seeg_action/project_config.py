import sys
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Qt5Agg")

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

p = Path().absolute()
this.current_subject = None
this.project_path = p.parent
#this.project_path = Path('/Users/senaer/Codes/CCNLab/sEEG-action-classes/')
this.data_path = this.project_path / 'data'
this.patients_path = this.data_path / 'subjects'
this.subject_path = None
this.steps_save_path = this.data_path / 'steps'
this.results_path = this.project_path / 'results'
this.exec_log_path = this.project_path / 'log'
this.raw_data_path = None
this.figure_save_path = this.project_path / 'results' / 'figures'
this.raw_fif_save_file = None
this.filtered_raw_file = None
this.epochs_file = None
this.bad_channels_file = None
this.ica_file = None
this.oct_6_src_file = None
this.vol_src_file = None
this.subject_head_mri_t = None
this.covariance_mat_file = None


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


def init_config(name):
    if this.current_subject is None:
        this.current_subject = name

        # Local
        if p.owner() == 'senaer':
            this.project_path = Path('/Users/senaer/Codes/CCNLab/sEEG-action-classes/')

        # Server
        elif p.owner() == 'ser':
            this.project_path = Path('/auto/k2/ser/sEEG-action-classes')

        this.data_path = this.project_path / 'data'
        this.patients_path = this.data_path / 'subjects'
        this.subject_path = this.patients_path / this.current_subject
        this.derivatives_path = this.subject_path / 'derivatives'
        # this.steps_save_path = this.data_path / 'steps'
        # this.results_path = this.project_path / 'results'
        this.exec_log_path = this.project_path / 'log'
        this.raw_data_path = this.subject_path / 'ses-01' / 'ieeg'
        this.figure_save_path = this.project_path / 'results' / 'figures'
        this.raw_fif_save_file = this.derivatives_path / f'{this.current_subject}-raw.fif.gz'
        this.filtered_raw_file = this.derivatives_path / f'{this.current_subject}-filtered-raw.fif.gz'
        this.epochs_file = this.derivatives_path / f'{this.current_subject}-epo.fif.gz'
        this.bad_channels_file = this.derivatives_path / f'{this.current_subject}-bad-channels.txt'
        this.ica_file = this.derivatives_path / f'{this.current_subject}-ica.fif.gz'
        this.montage_file = this.derivatives_path / f'{this.current_subject}-montage.fif'
        this.oct_6_src_file = this.subject_path / 'src' / f'{this.current_subject}-oct-6-src.fif.gz'
        this.vol_src_file = this.subject_path / 'src' / f'{this.current_subject}-vol-src.fif.gz'
        this.subject_head_mri_t = this.derivatives_path / f'{this.current_subject}-subject-mri-head-trans.fif'
        this.covariance_mat_file = this.derivatives_path / f'{this.current_subject}-cov.fif'





    else:
        raise UserWarning(f'Project is already configured! \n'
                          f'You have tried to set the current subject to "{name}". \n'
                          f'However, current subject is already set to {this.current_subject}')


this.event_id_to_code = {
    1: 'IP/ST',
    2: 'IP/DC',
    3: 'IP/SC',
    4: 'MN/ST',
    5: 'MN/DC',
    6: 'MN/SC',
    7: 'SD/ST',
    8: 'SD/DC',
    9: 'SD/SC'}
this.event_code_to_id = {v: k for k, v in this.event_id_to_code.items()}
this.condition_color_dict = \
    {1: 'red',
     2: 'lightcoral',
     3: 'rosybrown',
     4: 'blue',
     5: 'lightblue',
     6: 'lightsteelblue',
     7: 'springgreen',
     8: 'palegreen',
     9: 'darkseagreen',
     -1: 'black'
     }
