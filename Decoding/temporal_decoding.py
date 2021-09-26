import h5py
import numpy as np

# Mapping from condition to label
from sklearn import svm
from sklearn.model_selection import permutation_test_score


def decode_action_class(x, y, clf=svm.SVC(kernel='linear')):
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
    clf.fit(x, y)

    score, p_value = permutation_test_score(clf, x, y, scoring="accuracy", n_permutations=3, n_jobs=1)

    return score, p_value


def process_action_class_combinations(node, ac1, ac2):
    pow1 = node[ac1][:]
    pow2 = node[ac2][:]

    x = np.append(pow1, pow2, axis=0)
    y = [ac1] * len(pow1) + [ac2] * len(pow2)

    # classify
    score, p_value = decode_action_class(x, y)


def visitor_func(name, node):
    if isinstance(node, h5py.Group) and '/t_' in name:
        process_action_class_combinations(node, 'MN', 'IP')
        process_action_class_combinations(node, 'SD', 'IP')
        process_action_class_combinations(node, 'MN', 'SD')


if __name__ == '__main__':
    file = h5py.File('/Users/senaer/Codes/CCNLab/Intracerebral-ActionPerception/Decoding/power_19sep_13_30.hdf5', 'r')
    file.visititems(visitor_func)

