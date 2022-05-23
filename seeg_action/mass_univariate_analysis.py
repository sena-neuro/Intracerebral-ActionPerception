import mne
import numpy as np
from mne.stats import f_threshold_mway_rm, f_mway_rm
from mne.time_frequency import tfr_morlet
import patsy
import pandas as pd
from seeg_action import project_config as cfg
from mne.stats import linear_regression


def _prepare_lfp_data(epochs):
    data = np.array([epochs.get_data(item=condition)[:, 0, :] for condition in epochs.event_id])
    return data


def _prepare_power_data(epochs):
    freqs = np.geomspace(5, 152, num=50)  # define frequencies of interest
    n_cycles = freqs / freqs[0]

    epochs_power = list()
    for condition in [epochs[k] for k in epochs.event_id]:
        this_tfr = tfr_morlet(condition, freqs, n_cycles=n_cycles,
                              average=False, return_itc=False, n_jobs=-2)
        this_tfr.apply_baseline(mode='zlogratio', baseline=(None, 0))
        this_power = this_tfr.data[:, 0, :, :]  # we only have one channel.
        epochs_power.append(this_power)
    return np.asarray(epochs_power)


def mass_univariate_rm_anova(epochs, effect, input_type):
    if effect == 'action_class':
        effects = 'A'
    elif effect == 'stimulus_type':
        effects = 'B'
    elif effect == 'interaction':
        effects = 'A:B'

    factor_levels = [3, 3]

    def stat_fun(*args):
        return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                         effects=effects, return_pvals=False)[0]

    if input_type == 'lfp':
        data = _prepare_lfp_data(epochs)
    elif input_type == 'power':
        data = _prepare_power_data(epochs)

    # The ANOVA returns a tuple f-values and p-values, we will pick the former.
    pthresh = 0.05  # set threshold rather high to save some time
    n_replications = data.shape[1]

    f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effects,
                                   pthresh)
    tail = 1  # f-test, so tail > 0
    n_permutations = 1024
    F_obs, clusters, cluster_p_values, _ = mne.stats.permutation_cluster_test(
        data, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=-2,
        n_permutations=n_permutations, buffer_size=1000, out_type='indices')

    return F_obs, clusters, cluster_p_values


def regression(epochs):
    contrasts = ['action_vs_control',
                 'static_vs_dynamic',
                 'MN_vs_IP+SD',
                 'IP_vs_SD']

    design = pd.DataFrame({'condition': np.vectorize(cfg.event_id_to_code.get)(epochs.events[..., -1])})
    design['action_vs_control'] = np.where(design.condition.str.contains('ST'), 2, -3)
    design['static_vs_dynamic'] = np.where(design.condition.str.contains('SC'), 1,  # Static control
                                            np.where(design.condition.str.contains('DC'), -1,  # Dynamic control
                                                     0))  # else action stimulus condition
    design['MN_vs_IP+SD'] = np.where(design.condition.str.contains('MN'), 2,  # Manipulative actions
                                      np.where(
                                          (design.condition.str.contains('IP') | design.condition.str.contains('SD')),
                                          -1,
                                          0))  # else control stimuli
    design['IP_vs_SD'] = np.where(design.condition.str.contains('IP'), 1,  # Interpersonal actions
                                   np.where(design.condition.str.contains('SD'), -1,  # Skin-displacing actions
                                            0))
    design.drop(columns='condition', inplace=True)
    return mne.stats.linear_regression(epochs, design_matrix=design, names=contrasts)
