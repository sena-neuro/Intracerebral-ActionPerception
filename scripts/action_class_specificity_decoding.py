# %%
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedShuffleSplit
from itertools import combinations
from seeg_action.project_config import ProjectConfig as cfg
import seeg_action.preprocessing as pp
import mne
import numpy as np
from scipy.stats import zmap, trimboth
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%
subject = 'PetriceanuC'
cfg.init_current_subject(subject)

# %%
if not cfg.bipolar_ref_epochs_file.exists():
    pp.export_bipolar_reference_epochs()
epochs = mne.read_epochs(cfg.bipolar_ref_epochs_file)
gm_channels = pp.get_gm_channels(epochs.info)
epochs.pick(gm_channels)

epochs_action = epochs['ST']
for cond in epochs_action.event_id:
    action_class, _, action_examplar, object_size, actor = cond.split('/')
    evk_cs = epochs['/'.join([action_class, 'CS', object_size, actor])].average()
    evk_cd = epochs['/'.join([action_class, 'CD', object_size, actor])].average()
    evk_control = mne.combine_evoked([evk_cs, evk_cd], [0.5, 0.5])
    epochs_action[cond].subtract_evoked(evk_control)


epochs = epochs_action.copy()
epochs = epochs.apply_function(lambda x: zmap(x, trimboth(x, 0.1)))

# %%
freqs = np.geomspace(5, 152, num=50)
n_cycles = freqs / freqs[0]
tfr_tmin, tfr_tmax = -0.3, 2.3


# %%
df = pd.DataFrame()
df[['action_class', 'presented_stimulus', 'action_exampler', 'target_size', 'actor']] = pd.Series(
    epochs.events[..., -1]).apply(cfg.event_id_to_description.get).str.split('/', expand=True)

y = df.action_class.values
labels = sorted(list(set(y)))


# %%
ch = "N10-N11"
ch_epc = epochs.copy().pick(ch)

epochs_tfr = mne.time_frequency.tfr_morlet(
    ch_epc, freqs, n_cycles=n_cycles,
    average=False, return_itc=False, n_jobs=-3, verbose=False)
epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.3, -0.1), verbose=False)
epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)
X = epochs_tfr.data.squeeze()

# %%
pipe = make_pipeline(
    mne.decoding.Vectorizer(),
    PCA(n_components=.99, whiten=True),
    LinearDiscriminantAnalysis()
)

# %%
# def confusion_matrix_scorer(clf, X, y):
#     from sklearn.metrics import confusion_matrix
#     y_pred = clf.predict(X)
#     labels = sorted(list(set(y)))
#     cm = confusion_matrix(y, y_pred, labels=labels, normalize=None)
#     cm_labels = [f'{cond1}/{cond2}' for cond1 in labels for cond2 in labels]
#     return dict(zip(cm_labels, cm.ravel()))


cm_labels = [f'{cond1}/{cond2}' for cond1 in labels for cond2 in labels]

loo = LeaveOneOut()
lda = LinearDiscriminantAnalysis()

# Vectorizes
# (192, 50, 2601) -> (192, 130050)
X_vec = mne.decoding.Vectorizer().fit_transform(X)
# (192, 130050) -> (192, 32)
X_pca = PCA(n_components=.99, whiten=True).fit_transform(X_vec)

y_true, y_pred = [], []
for train_index, test_index in loo.split(X_pca):
    X_train, X_test, y_train, y_test = X_pca[train_index], X_pca[test_index], y[train_index], y[test_index]
    lda.fit(X_train, y_train)
    yhat = lda.predict(X_test)
    y_true.append(y_test[0])
    y_pred.append(yhat[0])

cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=None) # 'true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels
                              )
# im_kw=dict(vmin=10, vmax=32)
disp.plot(cmap=plt.cm.OrRd, im_kw=dict(vmin=0, vmax=50))

# %%
cv = StratifiedShuffleSplit(n_splits=30, test_size=10, random_state=42)

# %%
score_traces = dict()
for label_pair in combinations(labels, 2):
    idx  = np.isin(y, label_pair)
    y_cond, X_cond = y[idx], X[idx]
    time_decod = mne.decoding.SlidingEstimator(
        pipe,
        n_jobs=-3,
        scoring='roc_auc',
        verbose=False)

    scores_ = mne.decoding.cross_val_multiscore(
        time_decod,
        X_cond, y_cond,
        cv=cv,
        n_jobs=-3,
        verbose=False
    )

    # Mean scores across cross-validation splits
    score_traces['/'.join(label_pair)] = scores_

# %%
times = epochs_tfr.times
fig, axes = plt.subplots(nrows=3, ncols=1)
for ax, (label_pair, scores_) in zip(axes, score_traces.items()):
    # Mean scores across cross-validation splits
    scores = np.mean(scores_, axis=0)
    std_scores = np.std(scores_, axis=0)

    hyp_limits = (scores - std_scores, scores + std_scores)
    ax.fill_between(times, hyp_limits[0], y2=hyp_limits[1], alpha=0.1)
    ax.plot(times, scores, label='score')
    ax.axhline(.50, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('ROC-AUC')
    ax.legend()
    ax.axvline(0, color='k', linestyle='-')
    ax.set_title(f'{label_pair} decoding score')

# %%
