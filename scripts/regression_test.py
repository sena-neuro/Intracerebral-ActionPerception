import mne
from seeg_action import project_config
from seeg_action import preprocessing as pp
import numpy as np
import pandas as pd
import statsmodels.api as sm

# %%
project_config.init_current_subject('PriviteraM')
cfg = project_config.ProjectConfig

epochs = mne.read_epochs(cfg.epochs_file)
epochs.pick_types(seeg=True)
epochs.drop_channels(pp.get_wm_channels(epochs.info))

# %%
# Design matrix
design = pd.DataFrame({'condition': np.vectorize(cfg.event_id_to_description.get)(epochs.events[..., -1])})

design['action_vs_control'] = np.where(design.condition.str.contains('ST'), 2, -3)
design['static_vs_dynamic'] = np.where(design.condition.str.contains('CS'), 1,  # Static control
                                       np.where(design.condition.str.contains('CD'), -1,  # Dynamic control
                                                0))  # else action stimulus condition
design['MN_vs_IP+SD'] = np.where(design.condition.str.contains('OBJ/ST'), 2,  # Manipulative actions
                                 np.where(
                                     (design.condition.str.contains('PER/ST') | design.condition.str.contains(
                                         'BOD/ST')), -1,
                                     0))  # else control stimuli
design['IP_vs_SD'] = np.where(design.condition.str.contains('PER/ST'), 1,  # Interpersonal actions
                              np.where(design.condition.str.contains('BOD/ST'), -1,  # Skin-displacing actions
                                       0))
design.drop(columns='condition', inplace=True)
design = sm.add_constant(design)

# %%
freqs = np.geomspace(5, 152, num=50)
n_cycles = freqs / freqs[0]
tfr_tmin, tfr_tmax = -0.3, 2.3

channel = "V'9"
# for channel in epochs.ch_names:
epochs_ch = epochs.copy().pick(channel)

epochs_tfr = mne.time_frequency.tfr_morlet(epochs_ch, freqs, n_cycles=n_cycles,
                                           average=False, return_itc=False, n_jobs=-3)
epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0))
# epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)
epochs_tfr.crop(tmin=0., tmax=1.)


# %%
condition_contrast_matrix = {
    "OBJ/ST": np.array([1, 2, 0, 2, 0]),
    "OBJ/CS": np.array([1, -3, 1, 0, 0]),
    "OBJ/CD": np.array([1, -3, -1, 0, 0]),
    "PER/ST": np.array([1, 2, 0, -1, 1]),
    "PER/CS": np.array([1, -3, 1, 0, 0]),
    "PER/CD": np.array([1, -3, -1, 0, 0]),
    "BOD/ST": np.array([1, 2, 0, -1, -1]),
    "BOD/CS": np.array([1, -3, 1, 0, 0]),
    "BOD/CD": np.array([1, -3, -1, 0, 0])
}
contrast_vectors = {
    # ?
    # "static control > baseline": np.array([0, -3, 1, 0, 0]),
    # "dynamic control > baseline": np.array([0, -3, -1, 0, 0]),
    #"action > control": np.array([
    #    condition_contrast_matrix['PER/ST'] - condition_contrast_matrix['PER/SC'],
    #    condition_contrast_matrix['PER/ST'] - condition_contrast_matrix['PER/DC']
    #]),
    "OBJ-specificity": np.array([
        condition_contrast_matrix['OBJ/ST'] - condition_contrast_matrix['PER/ST'],
        condition_contrast_matrix['OBJ/ST'] - condition_contrast_matrix['BOD/ST']
    ]),
    "BOD-specificity": np.array([
        condition_contrast_matrix['BOD/ST'] - condition_contrast_matrix['PER/ST'],
        condition_contrast_matrix['BOD/ST'] - condition_contrast_matrix['PER/ST']
    ]),
    "PER-specificity": np.array([
        condition_contrast_matrix['PER/ST'] - condition_contrast_matrix['OBJ/ST'],
        condition_contrast_matrix['PER/ST'] - condition_contrast_matrix['BOD/ST']
    ]),
}


# %%# %%
n_trials, n_regressors = design.shape
data = epochs_tfr.data
_, n_channels, n_freqs, n_times = data.shape
data = np.reshape(data, (n_trials, -1))

regressor_coefs = np.empty((n_regressors, data.shape[-1]))
for i in range(data.shape[-1]):
    fitted_model = sm.OLS(data[:, i], design).fit()
    regressor_coefs[:, i] = fitted_model.params


# %%
condition_coefs = {condition: cvec @ regressor_coefs for condition, cvec in condition_contrast_matrix.items()}

# %%
from nilearn.mass_univariate import permuted_ols




# %%
# def func(arr):
#     return linalg.lstsq(a=design_matrix, b=arr)[0]
#
# est_beta = np.apply_along_axis(func, 0, y)

# %%

# %%
# betas, resid_sum_squares, _, _ = linalg.lstsq(a=design_matrix, b=y)
# beta = {predictor: x.reshape(data.shape[1:]) for x, predictor in zip(betas, names)}
# regressor_coefs = dict()
# for predictor in model.exog_names:
#    regressor_coefs[predictor] = mne.time_frequency.AverageTFR(
#        epochs_ch.info,
#        data.reshape((1, data.shape[0], data.shape[1])),
#        epochs_tfr.times, freqs, nave=64,
#        comment=predictor, method=None, verbose=None)
# %%

# # %%
# beta_maps = {
#     contrast:
#         mne.combine_evoked([res[reg].beta for reg in regressors], c_vec)
#     for contrast, c_vec in contrasts.items()
#  }
#
# # %%
#
# src = mne.read_source_spaces(cfg.oct_6_src_file)
# src_fs = mne.read_source_spaces(cfg.patients_path / 'fsaverage' / 'bem' / 'fsaverage-ico-5-src.fif')
#
#
# # %%
# def calculate_stcs(evks):
#     subject_stc = {}
#     fsaverage_stc = {}
#     for cond, evk in evks.items():
#         subject_stc[cond] = mne.stc_near_sensors(
#             evk, cfg.subject_head_mri_t, cfg.current_subject, subjects_dir=cfg.patients_path, src=src,
#             verbose='error')  # ignore missing electrode warnings
#
#         fsaverage_stc[cond] = mne.compute_source_morph(
#             subject_stc[cond], cfg.current_subject, 'fsaverage', cfg.patients_path,
#             smooth=5, verbose='error').apply(subject_stc[cond])
#     return subject_stc, fsaverage_stc
#
#
# # %%
# subject_stc, fsaverage_stc = calculate_stcs(beta_maps)
#
# # %%
# flat_brain = fsaverage_stc['PER_contrast'].plot(
#         surface='flat', hemi='split', subjects_dir=cfg.patients_path,
#         view_layout='vertical', time_viewer=True)

# %%

# %%

# %%
# betas, resid_sum_squares, _, _ = linalg.lstsq(a=design_matrix, b=y)
#
# %%
# n_rows, n_predictors = design_matrix.shape
# df = n_rows - n_predictors
# sqrt_noise_var = np.sqrt(resid_sum_squares / df).reshape(data.shape[1:])
# design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
# unscaled_stderrs = np.sqrt(np.diag(design_invcov))
# tiny = np.finfo(np.float64).tiny
# beta, stderr, t_val, p_val, mlog10_p_val = (dict() for _ in range(5))
# for x, unscaled_stderr, predictor in zip(betas, unscaled_stderrs, names):
#     beta[predictor] = x.reshape(data.shape[1:])
#     stderr[predictor] = sqrt_noise_var * unscaled_stderr
#     p_val[predictor] = np.empty_like(stderr[predictor])
#     t_val[predictor] = np.empty_like(stderr[predictor])
#
#     stderr_pos = (stderr[predictor] > 0)
#     beta_pos = (beta[predictor] > 0)
#     t_val[predictor][stderr_pos] = (beta[predictor][stderr_pos] /
#                                     stderr[predictor][stderr_pos])
#     cdf = stats.t.cdf(np.abs(t_val[predictor][stderr_pos]), df)
#     p_val[predictor][stderr_pos] = np.clip((1. - cdf) * 2., tiny, 1.)
#     # degenerate cases
#     mask = (~stderr_pos & beta_pos)
#     t_val[predictor][mask] = np.inf * np.sign(beta[predictor][mask])
#     p_val[predictor][mask] = tiny
#     # could do NaN here, but hopefully this is safe enough
#     mask = (~stderr_pos & ~beta_pos)
#     t_val[predictor][mask] = 0
#     p_val[predictor][mask] = 1.
#     mlog10_p_val[predictor] = -np.log10(p_val[predictor])

# %%
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 8))
#
# for ax, (predictor, data) in zip(axes.flatten(), beta.items()):
#
#
#
#     regressor_coefs = mne.time_frequency.AverageTFR(epochs_ch.info,
#                                              data.reshape((1, data.shape[0], data.shape[1])),
#                                              epochs_tfr.times, freqs, 64,
#                                              comment=predictor, method=None, verbose=None)
#     regressor_coefs.plot(title= predictor, axes=ax, vmin=-3, vmax=3, combine=None, show=False)
#     ax.set_title(f'{channel} {predictor}')

# %%

# %%
# res = mne.stats.linear_regression(epochs, design_matrix, names)
#
#
# def func(arr):
#     return linalg.lstsq(a=design_matrix, b=arr)[0]
#
# est_beta = np.apply_along_axis(func, 0, y)
