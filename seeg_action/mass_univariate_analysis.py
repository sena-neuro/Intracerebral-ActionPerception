import mne
import numpy as np
from mne.stats import f_threshold_mway_rm, f_mway_rm
from mne.time_frequency import tfr_morlet
import patsy
import pandas as pd
from seeg_action import project_config as cfg
from mne.stats import linear_regression
# pip install git+https://github.com/john-veillette/mne_ari.git
from mne_ari import all_resolutions_inference


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


def regression(epochs, input_type='power', correction='ari'):

    if input_type == 'lfp':
        data = epochs.get_data()
    elif input_type == 'power':
        freqs = np.geomspace(5, 152, num=50)  # define frequencies of interest
        n_cycles = freqs / freqs[0]
        epochs_tfr = tfr_morlet(epochs, freqs, n_cycles=n_cycles,
                              average=False, return_itc=False, n_jobs=-2)
        epochs_tfr.apply_baseline(mode='zlogratio', baseline=(None, 0))
        data = epochs_tfr.data

    # Design matrix
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


    beta, stderr, t_val, p_val, mlog10_p_val = _regression(data, design, contrasts)

    # Correction
    alpha = .05

    def regression_stat_fun(data):
        return p_val

    if correction == 'ari':
        # All resolutions interface
        # https://github.com/john-veillette/mne-ari

        # a function that take a (n_observations, n_tests) array for a one-sample test or
        # and returns an (n_tests, ) long array of p-values


        p_vals, tdp, clusters = all_resolutions_inference(data, alpha=alpha,
                                                          ari_type='permutation',
                                                          statfun=regression_stat_fun
                                                          )
        print('We found %d clusters' % len(clusters))
        return p_vals, tdp, clusters

    #elif correction == 'fdr':
    #    mask = []
    #    for i in len(contrasts):
    #        mask[i], pval_corrected[i] = mne.stats.fdr_correction(p_val, alpha=alpha)
    #    return (pval_corrected, mask)





def _regression(data, design_matrix, names):
    from scipy import stats, linalg

    n_samples = len(data)    # 576
    n_features = np.product(data.shape[1:]) # 3101

    n_rows, n_predictors = design_matrix.shape # (576, 4)

    y = np.reshape(data, (n_samples, n_features)) # (576, 1, 3101) -> (576, 3101)
    betas, resid_sum_squares, _, _ = linalg.lstsq(a=design_matrix, b=y)

    df = n_rows - n_predictors
    sqrt_noise_var = np.sqrt(resid_sum_squares / df).reshape(data.shape[1:])
    design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))
    tiny = np.finfo(np.float64).tiny
    beta, stderr, t_val, p_val, mlog10_p_val = (dict() for _ in range(5))
    for x, unscaled_stderr, predictor in zip(betas, unscaled_stderrs, names):
        beta[predictor] = x.reshape(data.shape[1:])
        stderr[predictor] = sqrt_noise_var * unscaled_stderr
        p_val[predictor] = np.empty_like(stderr[predictor])
        t_val[predictor] = np.empty_like(stderr[predictor])

        stderr_pos = (stderr[predictor] > 0)
        beta_pos = (beta[predictor] > 0)
        t_val[predictor][stderr_pos] = (beta[predictor][stderr_pos] /
                                        stderr[predictor][stderr_pos])
        cdf = stats.t.cdf(np.abs(t_val[predictor][stderr_pos]), df)
        p_val[predictor][stderr_pos] = np.clip((1. - cdf) * 2., tiny, 1.)
        # degenerate cases
        mask = (~stderr_pos & beta_pos)
        t_val[predictor][mask] = np.inf * np.sign(beta[predictor][mask])
        p_val[predictor][mask] = tiny
        # could do NaN here, but hopefully this is safe enough
        mask = (~stderr_pos & ~beta_pos)
        t_val[predictor][mask] = 0
        p_val[predictor][mask] = 1.
        mlog10_p_val[predictor] = -np.log10(p_val[predictor])

    return beta, stderr, t_val, p_val, mlog10_p_val
