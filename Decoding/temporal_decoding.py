from itertools import combinations

import h5py
import numpy as np

# Mapping from condition to label
from sklearn import svm
from sklearn.model_selection import permutation_test_score

event_code_to_action_category_map = {
    "1": "SD",      # Skin-displacing
    "4": "MN",      # Manipulative
    "7": "IP",      # Interpersonal
}
action_categories = [ac for ac in event_code_to_action_category_map.values()]


def binary_action_category_decoder(x, y, clf=svm.SVC(kernel='linear')):
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
    clf.fit(x, y)
    score = None
    p_value = None

    # score, perm_scores, p_value = permutation_test_score(
    #    clf, x, y, scoring="accuracy", n_permutations=3, n_jobs=1)

    return score, p_value


file = h5py.File('/Users/senaer/Codes/CCNLab/Intracerebral-ActionPerception/Decoding/power_19sep_13_30.hdf5', 'r')

for subject in file:
    for lead in file[subject]:
        for time in file[subject][lead]:
            group = file[subject][lead][time]

            for ac_pair in combinations(action_categories, 2):
                print('ac_pair: ', ac_pair)
                pow1 = group[ac_pair[0]][:].transpose()
                pow2 = group[ac_pair[1]][:].transpose()
                # sd_pow = group['SD'][:].transpose()
                x = np.append(pow1, pow2, axis=0)
                y = [ac_pair[0]] * len(pow1) + [ac_pair[1]] * len(pow2)

                # classify
                score, p_value = binary_action_category_decoder(x, y)
                print(score, ', ', p_value)

