import h5py
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from pathlib import Path


parent_path = Path('/auto/data2/oelmas/Intracerebral')
output_path = parent_path / 'Results'

decoding_results_hdf_file = str(output_path / 'TEST_decoding_results.hdf5')

def decode_action_class(x, y):
    """
    Pairwise action category decoding

    :rtype: float, array (shape: permutation), float
    :returns score, perm_scores, p_value
    :keyword x : Any
        power
    :keyword y : Any
        action category
    """
    # Permutation test
    # param_grid=param_grid, n_jobs=1
    # Classifier for the decoding
    clf = Pipeline([('scale', StandardScaler()),
                    ('clf', svm.SVC(kernel='linear', C=1e-04))])

    # Create Cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    clf.fit(x, y)

    score, perm_scores, p_value = permutation_test_score(clf, x, y, scoring="accuracy", n_permutations=3, n_jobs=1)
    return score, p_value


def process_action_class_combinations(node, ac1, ac2):
    pow1 = node[ac1][:]
    pow2 = node[ac2][:]

    x = np.append(pow1, pow2, axis=0)
    y = [ac1] * len(pow1) + [ac2] * len(pow2)

    # classify
    return decode_action_class(x, y)


def visitor_func(name, node):
    if isinstance(node, h5py.Group) and '/t_' in name:

        mn_ip_res = process_action_class_combinations(node, 'MN', 'IP')
        sd_ip_res = process_action_class_combinations(node, 'SD', 'IP')
        mn_sd_res = process_action_class_combinations(node, 'MN', 'SD')

        with h5py.File(decoding_results_hdf_file, 'a') as f:
            group = f.create_group(name=name)
            group.create_dataset(name='MNvsIP', data=mn_ip_res)
            group.create_dataset(name='SDvsIP', data=sd_ip_res)
            group.create_dataset(name='MNvsSD', data=mn_sd_res)


if __name__ == '__main__':
    with  h5py.File('/auto/data2/oelmas/Intracerebral/Data/BENEDETTI_TEST_power_data.hdf5', 'r') as file:
    	file.visititems(visitor_func)
