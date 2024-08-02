import numpy as np
import mne
import pandas as pd
from seeg_action import project_config as cfg
from scipy import stats, linalg


def regression_power(epochs):
    conds = list(set(key[:6] for key in epochs.event_id.keys()))
    design = pd.DataFrame({cond: np.char.startswith(
        np.vectorize(cfg.event_id_to_description.get)(epochs.events[..., -1]), cond).astype('int')
                           for cond in conds})
    design['intercept'] = 1

    names = design.columns
    design_matrix = design.to_numpy()

    freqs = np.geomspace(5, 152, num=50)
    n_cycles = freqs / freqs[0]
    tfr_tmin, tfr_tmax = -0.3, 2.3

    epochs_tfr = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles=n_cycles,
                                               average=False, return_itc=False, n_jobs=-3)
    epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0))
    epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)

    data = epochs_tfr.data

    n_samples = len(data)  # 576
    n_features = np.product(data.shape[1:])  # 3101

    n_rows, n_predictors = design_matrix.shape  # (576, 4)

    y = np.reshape(data, (n_samples, n_features))  # (576, 1, 3101) -> (576, 3101)
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






