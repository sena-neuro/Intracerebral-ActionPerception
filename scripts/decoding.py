# %%
import mne
import sklearn.model_selection

from seeg_action.project_config import ProjectConfig as config
from seeg_action import preprocessing as pp
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# %%
config.init_current_subject('PetriceanuC')
mne.set_log_file('ERROR')

# %%
epochs = mne.read_epochs(config.epochs_file)

# %%
#print(pp.get_wm_channels(epochs.info))
#epochs.drop_channels(pp.get_wm_channels(epochs.info))

# %%
epochs.pick_types(seeg=True)
epochs = epochs['ST']

# %%
# TFR parameters
freqs = np.geomspace(5, 152, num=50)
n_cycles = freqs / freqs[0]
tfr_tmin, tfr_tmax = -0.3, 2.3

# Classifier
clf = make_pipeline(
    mne.decoding.Vectorizer(),
    PCA(),
    LinearDiscriminantAnalysis()
)

# Label of each trial
y = np.vectorize(config.event_id_to_description.get)(epochs.events[..., -1]).astype('U6')
labels = sorted(list(set(y)))

n_time = int(tfr_tmax * epochs.info['sfreq'] - (tfr_tmin * epochs.info['sfreq']) + 1)
channels = epochs.ch_names
n_channels = len(channels)
f1_score_time_traces = {label: np.empty((n_channels, n_time))
                        for label in labels}

for ch_idx in range(n_channels):
    # Calculate power
    epochs_tfr = mne.time_frequency.tfr_morlet(
        epochs, freqs, n_cycles=n_cycles,
        picks=ch_idx, average=False, return_itc=False, n_jobs=-3)
    epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0))
    epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)
    times = epochs_tfr.times

    # Decoding
    X = epochs_tfr.data

    for cond in labels:
        time_decod = mne.decoding.SlidingEstimator(
            clf,
            n_jobs=-3,
            scoring=make_scorer(
                sklearn.metrics.f1_score, average=None, labels=[cond]),
            verbose=False)

        scores_ = mne.decoding.cross_val_multiscore(
            time_decod,
            X, y,
            cv=10,
            n_jobs=-3)

        # Mean scores across cross-validation splits
        f1_score_time_traces[cond][ch_idx, :] = np.mean(scores_, axis=0)

# %%
evokeds = []
for cond, score in f1_score_time_traces.items():
    nave = np.count_nonzero(y == cond)
    evokeds.append(
        mne.EvokedArray(
            data=score,
            info=epochs.info,
            tmin=tfr_tmin,
            comment=cond,
            nave=nave)
    )

# %%
mne.write_evokeds(config.f1_score_file, evokeds)

# %%
def plot_score_trace(score_evk):
    import matplotlib.pyplot as plt
    chance_level = 1. / len(score_evk)
    fig, axes = plt.subplots(3, 3)
    for ax, evk in zip(axes.ravel(), score_evk):
        ax.plot(evk.times, evk.data, label='score')
        ax.axhline(chance_level, color='k', linestyle='--', label='chance')
        ax.set_xlabel('Times')
        ax.set_ylabel('Score')
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')
        ax.set_title(evk.comment)
    fig.show()


# %%
def confusion_scorer(y_true, y_pred, cond1, cond2):
    def confusion_matrix_scorer(y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='all')
        cm_labels = [f'{cond1}/{cond2}' for cond1 in labels for cond2 in labels]
        return dict(zip(cm_labels, cm.ravel()))
    return confusion_matrix_scorer(y_true, y_pred,)[f'{cond1}/{cond2}']