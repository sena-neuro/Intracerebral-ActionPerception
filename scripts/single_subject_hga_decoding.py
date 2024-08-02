# %%
from seeg_action.project_config import ProjectConfig as cfg
import mne
import numpy as np
from scipy.stats import trim_mean, ttest_ind
from itertools import pairwise
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# %%
subject = 'PetriceanuC'
cfg.init_current_subject(subject)
epochs = mne.read_epochs(cfg.epochs_file)
# orig_epochs = epochs.copy()
epochs = epochs.pick_types(seeg=True)
# P3 bad in PetriceanuC
epochs.info['bads'].extend(['P3'])
epochs.apply_baseline((None, 0))


# %%
hga = epochs.copy().filter(50, 150).apply_hilbert(
    n_jobs=-3, envelope=True)
hga.crop(tmin=-0.4, tmax=2.5)
hga.apply_baseline((-0.4, 0))

# %%
conditions = ['/'.join((ac, st)) for ac in cfg.action_classes for st in cfg.presented_stimuli]

# %%
window_times = list(pairwise(np.arange(0., hga.tmax, 0.4)))
p_vals = np.empty((len(conditions), len(hga.ch_names), len(window_times)))
for cond_idx, cond in enumerate(conditions):
    for ch_idx, ch in enumerate(hga.ch_names):
        baseline_hga = trim_mean(hga.get_data(item=cond, tmin=-0.4, tmax=0., picks=[ch]), 0.1, axis=2).squeeze()
        for window_idx, window in enumerate(window_times):
            tmin, tmax = window
            window_hga = trim_mean(hga.get_data(item=cond, tmin=tmin, tmax=tmax, picks=[ch]), 0.1, axis=2).squeeze()
            t_stat, p_val = ttest_ind(baseline_hga, window_hga)
            p_vals[cond_idx][ch_idx][window_idx] = p_val

# %%
reject, corr_p_vals = mne.stats.fdr_correction(p_vals, alpha=.05)
responsive_channels = np.asarray(hga.ch_names)[np.any(reject, axis=(0, 2))]

# %%
responsive_channels = ['G11', 'G12', 'G13', 'G14', 'K1', 'K2', 'K6', 'N1', 'N8', 'N10',
                       'N11', 'N12', 'P9', 'P10', 'P12', 'P14', 'P15', "H'1",
                       "J'1", "K'1", "N'14", "P'9", "P'10", "X'6", "Z'6"]

# %%
# hga.plot_image(picks=responsive_channels)

# %%
fig, axes = plt.subplots(3, 3)

for cond, ax in zip(conditions, axes.ravel()):
    hga_evoked = hga[cond].average(method=lambda x: trim_mean(x, 0.1, axis=0))
    hga_evoked.plot_image(axes=ax, picks=responsive_channels)
    ax.set_title(cond)
    del hga_evoked

# %%
hga = hga['ST']
hga.pick(responsive_channels)

# %%
# Classifier
clf = make_pipeline(
    mne.decoding.Scaler(hga.info, scalings='median'),
    # mne.decoding.Vectorizer(),
    mne.decoding.CSP(n_components=10, rank='full'),
    LinearDiscriminantAnalysis()
)

# %%
X = hga.get_data()
# Label of each trial
y = np.vectorize(cfg.event_id_to_description.get)(hga.events[..., -1]).astype('U6')
labels = sorted(list(set(y)))

# %%
time_window_length = 20
stride = 5

X_cond = np.lib.stride_tricks.sliding_window_view(X, time_window_length, axis=-1)
X_cond = np.swapaxes(X_cond, -2, -1)[..., ::stride]

times = np.lib.stride_tricks.sliding_window_view(hga.times, time_window_length, axis=-1)
times = np.swapaxes(times, -2, -1)[..., ::stride]
times = times[int(time_window_length/2), :]

# %%
time_decod = mne.decoding.SlidingEstimator(
    clf,
    scoring='roc_auc',
    n_jobs=-3,
    verbose=False
)

scores_ = mne.decoding.cross_val_multiscore(
    time_decod,
    X_cond, y,
    cv=10,
    n_jobs=-3,
    verbose=False
)

# %%
# Mean scores across cross-validation splits
scores = np.mean(scores_, axis=0)
std_scores = np.std(scores_, axis=0)

# %%
fig, ax = plt.subplots()
hyp_limits = (scores - std_scores, scores + std_scores)
ax.fill_between(times, hyp_limits[0], y2=hyp_limits[1], color='b', alpha=0.1)
ax.plot(times, scores, label='score')
ax.axhline(.33, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('Accuracy')  # Area Under the Curve
ax.legend()
ax.axvline(0, color='k', linestyle='-')
ax.set_title(f'Decoding accuracy')