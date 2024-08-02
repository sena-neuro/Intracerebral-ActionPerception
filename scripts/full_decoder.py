# %%
import pandas as pd
from seeg_action.project_config import ProjectConfig as cfg
import seeg_action.preprocessing as pp
import mne
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import permutation_test_score, LeaveOneGroupOut

# %%
subject = 'FedericoR'
cfg.init_current_subject(subject)

# %%
if not cfg.bipolar_ref_epochs_file.exists():
    pp.export_bipolar_reference_epochs()
epochs = mne.read_epochs(cfg.bipolar_ref_epochs_file)
epochs.pick_types(seeg=True)
orig_epochs = epochs.copy()

# %%
epochs_action = orig_epochs['ST']
for cond in epochs_action.event_id:
    action_class, _, action_examplar, object_size, actor = cond.split('/')
    evk_cs = epochs['/'.join([action_class, 'CS', action_examplar, object_size, actor])].average()
    evk_cd = epochs['/'.join([action_class, 'CD', action_examplar, object_size, actor])].average()
    evk_control = mne.combine_evoked([evk_cs, evk_cd], [0.5, 0.5])
    epochs_action[cond].subtract_evoked(evk_control)

epochs = epochs_action.copy()

# %%
# from scipy.stats import zmap, trimboth
# Honestly, I don't think this is necessary since we do z-scoring after tfr
# epochs = epochs.apply_function(lambda x: zmap(x, trimboth(x, 0.1)))

# %%
freqs = np.geomspace(5, 152, num=50)
n_cycles = freqs / freqs[0]
# tfr_tmin, tfr_tmax = -0.3, 2.3
tfr_tmin, tfr_tmax = 0.0, 2.3

# %%
df = pd.DataFrame()
df[['action_class', 'presented_stimulus', 'action_exampler', 'target_size', 'actor']] = pd.Series(
    epochs.events[..., -1]).apply(cfg.event_id_to_description.get).str.split('/', expand=True)
le = LabelEncoder()
y = le.fit_transform(df.action_class)

# %%
groups = np.empty(y.shape, dtype=np.int32)
by_action_class = df.drop(columns='presented_stimulus').groupby('action_class')
for ac in cfg.action_classes:
    ac_group = by_action_class.get_group(ac)
    by_stimulus = ac_group.groupby(['action_exampler', 'target_size', 'actor'])
    for group, trial_idx in enumerate(by_stimulus.groups.values()):
        groups[trial_idx] = group

# %%
lda = LinearDiscriminantAnalysis()
cv = LeaveOneGroupOut()

# %%
permutation_scores = dict.fromkeys(epochs.ch_names)
for ch in epochs.ch_names:
    ch_epc = epochs.copy().pick(ch)

    epochs_tfr = mne.time_frequency.tfr_morlet(
        ch_epc, freqs, n_cycles=n_cycles,
        average=False, return_itc=False, n_jobs=-3, verbose=False)
    epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.3, -0.1), verbose=False)
    epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)

    X = epochs_tfr.data.squeeze()

    # Vectorizes
    # (192, 50, 2301) -> (192, 115050)
    X = mne.decoding.Vectorizer().fit_transform(X)
    # (192, 115050) -> (192, 111)
    X = PCA(n_components=.99, whiten=True).fit_transform(X)

    try:
        score, permutation_scores[ch], pvalue = permutation_test_score(
            lda, X, y,
            groups=groups,
            n_permutations=1000,
            cv=cv,
            scoring='accuracy',
            n_jobs=-3,
            verbose=False)

        if pvalue < .05:
            print(' - ', end='')
        print(f'{ch} accuracy: {score}, p-value: {pvalue}')
    except ValueError:
        print(f'Value Error! Skipping {ch}...')


# %%
import pickle

with open(cfg.permutation_scores_file, 'wb') as f:
    pickle.dump(permutation_scores, f)
