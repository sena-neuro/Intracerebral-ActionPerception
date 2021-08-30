from pathlib import Path
import pandas as pd
import datetime


def find_specificity(row):
    if row[0] != 'ns' and row[1] != 'ns' and row[2] != 'ns':
        return 'ALL'
    elif row[0] != 'ns' and row[2] != 'ns':
        return 'MN'
    elif row[1] != 'ns' and row[2] != 'ns':
        return 'SD'
    elif row[0] != 'ns' and row[1] != 'ns':
        return 'IP'
    else:
        return 'NONE'


parent_path = Path('/auto/data2/oelmas/Intracerebral')
input_path = parent_path / 'Results' / 'SubjectDecodingResults'
output_path = input_path
subject_pickles = [x for x in input_path.iterdir() if x.match('25*.pkl')]

date = datetime.datetime.today().strftime('%d-%m')

for s_pkl in subject_pickles:
    df_svm = pd.read_pickle(s_pkl)
    subject_name = df_svm['subject'][0]
    subject_name = str.split(subject_name)[0]

    df_svm_sig = df_svm.loc[lambda df: df.p_value < .05, :]
    df_svm_sig = df_svm_sig.pivot_table(values='accuracy',
                                        index=['subject', 'lead'],
                                        columns=['classification_type'],
                                        aggfunc='first',
                                        fill_value='ns')

    df_svm_sig['specificity'] = df_svm_sig.apply(lambda row: find_specificity(row), axis=1)

    filename = date + '_' + subject_name + '_sig_res_specificity.pkl'
    sig_results_pkl_file = output_path / filename
    df_svm_sig.to_pickle(sig_results_pkl_file)
