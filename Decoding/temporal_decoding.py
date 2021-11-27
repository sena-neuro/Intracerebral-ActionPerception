import datetime
import re
import h5py
import scipy
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from pathlib import Path
import mne

if Path().owner() == 'senaer':
    parent_path = Path('/Users/senaer/Codes/CCNLab/Intracerebral-ActionPerception')
else:
    parent_path = Path('/auto/data2/oelmas/Intracerebral')
input_path = parent_path / 'Data'
output_path = parent_path / 'Results'

hdf_file = input_path / 'intracerebral_action_data.hdf5'

date = datetime.datetime.today().strftime('%d-%m')
#output_f_name = date + '_decoding_results.hdf5'
#decoding_results_hdf_file = output_path / output_f_name

action_classes = ['MN', 'IP', 'SD']


def decode_action_class(x, y):
    """
    Action category decoding - one versus rest
    :keyword x
        power
    :keyword y
        action category

    :returns t_val_dict

    """
    # Permutation test
    # param_grid=param_grid, n_jobs=1
    # Classifier for the decoding
    clf = Pipeline([('scale', StandardScaler()),
                    ('clf', svm.SVC(kernel='linear', C=1e-04))])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    n_splits = cv.get_n_splits()
    t_val_dict = dict.fromkeys(action_classes, np.empty(n_splits, dtype='float64'))
    for split_idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        x_train, x_test, y_train, y_test = x[train_idx], x[test_idx], y[train_idx], y[test_idx]
        clf.fit(x_train, y_train)

        for ac_idx, action_class in enumerate(clf.classes_):
            # Trials belonging to a specific action class
            true_ac_trials_idx = np.where(y_test == action_class)
            rest_trials_idx = np.where(y_test != action_class)

            d_vals = clf.decision_function(x_test)  # Distance to MN, IP, SD hyperplanes
            # True ac trials' distance to the hyperplane of the true action class
            true_ac_d_val = d_vals[true_ac_trials_idx, ac_idx].flatten()
            # Rest of the trials' distance to the hyperplane of the true action class
            rest_d_val = d_vals[rest_trials_idx, ac_idx].flatten()

            t_val = scipy.stats.ttest_ind(true_ac_d_val, rest_d_val).statistic
            t_val_dict[action_class][split_idx] = t_val

    # Average splits
    t_val_dict = {key: np.nanmean(value) for key, value in t_val_dict.items()}

    return t_val_dict


def decode():
    mn_pow = hdf['MN/power'][:]
    ip_pow = hdf['IP/power'][:]
    sd_pow = hdf['SD/power'][:]

    x = np.concatenate((mn_pow, ip_pow, sd_pow), axis=0)
    y = np.array(['MN'] * len(mn_pow) + ['IP'] * len(ip_pow) + ['SD'] * len(ip_pow))

    # classify
    res = decode_action_class(x, y)

    with h5py.File(decoding_results_hdf_file, 'a') as f:
        subj_lead_group = f.require_group(name=subj_lead_key)
        for ac, t_val in res.items():
            subj_lead_ac_key = subj_lead_key + '/' + ac
            ac_group = subj_lead_group.require_group(name=subj_lead_ac_key)
            dset = ac_group.require_dataset(name="t-vals", shape=(200,), dtype=float)
            time_idx = int(name[search.end():])
            dset[time_idx] = t_val


def identity(x):
    return x


if __name__ == '__main__':

    with h5py.File(power_hdf_file, 'r') as power_hdf:
        for _, subj_group in power_hdf.items():
            for _, lead_group in subj_group.items():
                for _, time_group in lead_group.items():
                    decode(time_group.name, time_group)

                with h5py.File(decoding_results_hdf_file, 'a') as res_hdf:
                    for ac_name, ac_group in lead_group.items():
                        t_vals = ac_group["t-vals"][:].reshape((1,-1))
                        # T_obs, clusters, cluster_p_values, H0
                        p_thresh = 0.025    # Two-tailed
                        n_samples = t_vals.shape[1]
                        thresh = -scipy.stats.distributions.t.ppf(p_thresh, n_samples - 1)

                        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(t_vals,
                                                                                                         tail=0,
                                                                                                         threshold=thresh,
                                                                                                         stat_fun=identity)
                        res_list = [T_obs, clusters, cluster_p_values, H0]
                        dset = ac_group.create_dataset(name="cluster_stat_res", data=res_list)
