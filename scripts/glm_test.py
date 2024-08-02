# %%
from sklearn.preprocessing import OneHotEncoder

from seeg_action.project_config import ProjectConfig as cfg
import seeg_action.preprocessing as pp
import mne
import numpy as np
from numpy.linalg import inv

# %%
subject = 'PriviteraM'
cfg.init_current_subject(subject)

# %%
if not cfg.bipolar_ref_epochs_file.exists():
    pp.export_bipolar_reference_epochs()
epochs = mne.read_epochs(cfg.bipolar_ref_epochs_file)

# %%
y = epochs.get_data()
n_trials, n_channels, n_times = y.shape
# y.shape : (576, 480655)
y = y.reshape(n_trials, -1)

# %%
ohe = OneHotEncoder(sparse=False)  # we don't want a "sparse" output
X = ohe.fit_transform(epochs.events[..., -1][:, np.newaxis])
# X.shape : (576, 145)
X = np.concatenate([np.ones((n_trials, 1)), X], axis=1)

# %%
# betas.shape : (145, 480655)
betas = inv(X.T @ X) @ X.T @ y

# residuals.shape : (576, 480655)
residuals = y - (X  @ betas)

# %%
n_regressors = betas.shape[0]
c_mat = np.zeros((n_regressors, n_regressors))

for cond, value in epochs.event_id.items():
    action_class, presented_stimulus, action_examplar, object_size, actor = cond.split('/')
    if presented_stimulus == 'ST':
        st_code = epochs.event_id[cond]
        cs_cond = '/'.join([action_class, 'CS', action_examplar, object_size, actor])
        cd_cond = '/'.join([action_class, 'CD', action_examplar, object_size, actor])
        cs_code = epochs.event_id[cs_cond]
        cd_code = epochs.event_id[cd_cond]

        c_mat[value, st_code] = 1
        c_mat[value, cs_code] = -0.5
        c_mat[value, cd_code] = -0.5


# %%
# y.shape : (576, 480655)
# X.shape : (576, 145)
# c_vec : (145, 145)
# betas.shape : (145, 480655)
# residuals.shape : (576, 480655)

y_hat = (X @ c_mat @ betas) + residuals
y_hat = y_hat.reshape(n_trials, n_channels, n_times)

# %%
regressed_epochs = mne.EpochsArray(data=y_hat,
                                   info=epochs.info,
                                   events=epochs.events,
                                   tmin=epochs.tmin,
                                   event_id=epochs.event_id,
                                   baseline=epochs.baseline,
                                   )
regressed_epochs = regressed_epochs['ST']

# %%