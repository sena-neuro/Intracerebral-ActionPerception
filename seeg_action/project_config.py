import sys
from pathlib import Path
import argparse

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

this.current_subject = 'BerberiM'

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
    if this.current_subject_name is None:
        this.current_subject_name = name
    else:
        raise UserWarning(f'You have tried to set the current subject to "{name}". \n'
                          f'However, current subject is already set to {this.current_subject_name}')


# Local
if Path().owner() == 'senaer':
    this.project_path = Path('/Users/senaer/Codes/CCNLab/sEEG-action-classes/')
    this.data_path = this.project_path / 'data'
    this.patients_path = this.data_path / 'subjects'
    this.steps_save_path = this.data_path / 'steps'
    this.results_path = this.project_path / 'results'
    this.exec_log_path = this.project_path / 'log'
    this.bad_annots_save_path = this.exec_log_path / 'bad_electrodes'
    this.raw_data_path = this.patients_path / this.current_subject / 'ses-01' / 'ieeg'


# Server
elif Path.owner() == 'ser':
    # Constants
    this.raw_data_path = Path('/auto/data/burgen/StereoEEG_ActionBase')
    this.output_path = Path('/auto/data2/ser/StereoEEG_ActionBase')
    this.eeg_data_path = this.raw_data_path / 'EEGdata'
    this.event_codes_path = this.raw_data_path / 'LOG'
    this.patients_path = this.output_path / 'patients'
    this.steps_save_path = this.output_path / 'steps'
    this.bad_annots_save_path = this.steps_save_path / 'BAD electrodes'
    this.results_path = this.output_path / 'results'
    this.exec_log_path = this.output_path / 'log'

if not this.bad_annots_save_path.exists():
    this.bad_annots_save_path.mkdir()

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