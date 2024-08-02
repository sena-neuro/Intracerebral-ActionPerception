# %%
from seeg_action.project_config import ProjectConfig as cfg
import mne
import numpy as np
import pandas as pd
from mne.decoding import EMS, compute_ems
from scipy.stats import zmap, trimboth, trim_mean
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import re
import matplotlib.pyplot as plt

# %%
subject = 'PetriceanuC'
cfg.init_current_subject(subject)
conditions = ['/'.join((ac, st)) for ac in cfg.action_classes for st in cfg.presented_stimuli]

# %%
epochs = mne.read_epochs(cfg.bipolar_ref_epochs_file)

# %%
# flat: "Z'8-Z'9" flat = 1e-2
# totally bad: 'P2-P3', 'P3-P4', 'P4-P5', 'J1-J2', 'J2-J3', 'J3-J4'
# weird but not sure high oscillations: "N'11-N'12", "N'12-N'13"
epochs.info['bads'] = ['P2-P3', 'P3-P4', 'P4-P5', "Z'8-Z'9",
                       'J1-J2', 'J2-J3', 'J3-J4']

# %%
epochs = epochs.apply_function(lambda x: zmap(x, trimboth(x, 0.1)))

# %%
evk_dict = {}
# Does it take all data than trim or trim each cond than take mean?
for cond in conditions:
    evk_dict[cond] = epochs[cond].average(method=lambda x : trim_mean(x, 0.1))
    
# %%
for cond, evk in evk_dict.items():
    evk.plot(window_title=cond)

# %%
# shafts
pattern = r'^([A-Z]\'?)\d+'
shafts = list(set(re.search(pattern, ch).groups()[0] for ch in epochs.ch_names))

# %%
channels_by_shaft = {
    shaft :
    [ch_idx for ch_idx, ch in enumerate(epochs.ch_names) if re.search(pattern, ch).groups()[0] == shaft]
    for shaft in shafts
}

# %%
evk = evk_dict['PER/ST']

# %%
evk.plot_image(
    group_by=channels_by_shaft,
    clim=dict(seeg=(-0.8, 0.8)),
    scalings=dict(seeg=1),
    show_names='all',
    units=dict(seeg='z-scored voltage'))

# %%
mne.viz.plot_compare_evokeds(
    evk_dict,
    # picks=['G13-G14'],
    legend='lower right',
    split_legend=True,
    colors={'OBJ': 'tab:orange', 'PER': 'tab:blue', 'BOD': 'tab:green'},
    linestyles={'ST': 'solid', 'CS': 'dotted', 'CD': 'dashed'}
)

# %%
fig, axes = plt.subplots(3, 3)
for ax, (cond, evk) in zip(axes.ravel(), evk_dict.items()):
    evk.plot_image(axes=ax,
                   vmin=-0.7, vmax=0.7,
                   unit=True,
                   scalings=dict(seeg=1e-3),
                   units=dict(seeg='z-scored voltage')
                   )
    ax.set_title(cond)


# %%
labels = ['ST', 'CD']
epochs = epochs[labels]

# %%
df = pd.DataFrame()
df[['action_class', 'presented_stimulus', 'action_exampler', 'target_size', 'actor']] = pd.Series(epochs.events[..., -1]).apply(cfg.event_id_to_description.get).str.split('/', expand=True)

# %%
# Setup the data to use it a scikit-learn way:
X = epochs.get_data()
n_epochs, n_channels, n_times = X.shape

le = LabelEncoder()
y = le.fit_transform(df.presented_stimulus)

# %%
# Initialize EMS transformer
ems = EMS()

# Initialize the variables of interest
X_transform = np.zeros((n_epochs, n_times))  # Data after EMS transformation
filters = list()  # Spatial filters at each time point

# In the original paper, the cross-validation is a leave-one-out. However,
# we recommend using a Stratified KFold, because leave-one-out tends
# to overfit and cannot be used to estimate the variance of the
# prediction within a given fold.

for train, test in StratifiedKFold(n_splits=5).split(X, y):

    # In the original paper, the z-scoring is applied outside the CV.
    # However, we recommend to apply this preprocessing inside the CV.
    # Note that such scaling should be done separately for each channels if the
    # data contains multiple channel types.
    X_scaled = X / np.std(X[train])

    # Fit and store the spatial filters
    ems.fit(X_scaled[train], y[train])

    # Store filters for future plotting
    filters.append(ems.filters_)

    # Generate the transformed data
    X_transform[test] = ems.transform(X_scaled[test])

# Average the spatial filters across folds
filters = np.mean(filters, axis=0)

# %%
# Plot individual trials
plt.figure()
plt.title('single trial surrogates')
plt.imshow(X_transform[y.argsort()], origin='lower', aspect='auto',
           vmin=-3, vmax=3,
           extent=[epochs.times[0], epochs.times[-1], 1, len(X_transform)],
           cmap='RdBu_r')
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Trials (reordered by condition)')
plt.show()

# %%
y_inv = le.inverse_transform(y)

# Plot average response
plt.figure()
plt.title('Average EMS signal')
plt.legend(loc='best')
plt.show()

# %%
for label in labels:
    ems_ave = X_transform[y_inv == label]
    plt.plot(epochs.times, ems_ave.mean(0), label=label)
plt.xlabel('Time (ms)')
plt.ylabel('a.u.')
plt.legend(loc='best')
plt.show()

# %%
# Visualize spatial filters across time
evoked = mne.EvokedArray(filters, epochs.info, tmin=epochs.tmin)
evoked.plot_topomap(scalings=1)