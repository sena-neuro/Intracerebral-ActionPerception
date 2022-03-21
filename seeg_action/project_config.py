import sys
from pathlib import Path
import argparse

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# Constants
this.raw_data_path = Path('/auto/data/burgen/StereoEEG_ActionBase')
this.eeg_data_path = this.raw_data_path / 'EEGdata'
this.event_codes_path = this.raw_data_path / 'LOG'
this.output_path = Path('/auto/data2/ser/StereoEEG_ActionBase')
this.steps_save_path = this.output_path / 'steps'
this.results_path = this.output_path / 'results'
this.exec_log_path = this.output_path / 'log'
this.event_id_to_code = {
    1: 'IP-ST',
    2: 'IP-DC',
    3: 'IP-SC',
    4: 'MN-ST',
    5: 'MN-DC',
    6: 'MN-SC',
    7: 'SD-ST',
    8: 'SD-DC',
    9: 'SD-SC'}
this.event_code_to_id = {v: k for k, v in this.event_id_to_code.items()}

# Module-wide session variables
module_wide_variables = {
    'current_subject': None,
    'redo': False
}


def set_var(variable_name, variable_value):
    this.module_wide_variables[variable_name] = variable_value


def get_var(variable_name):
    return this.module_wide_variables[variable_name]


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
