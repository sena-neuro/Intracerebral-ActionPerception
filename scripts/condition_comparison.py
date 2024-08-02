# %%
from seeg_action.project_config import ProjectConfig as cfg
import mne
from scipy.stats import zmap, trimboth, trim_mean
import numpy as np
import matplotlib.pyplot as plt

# %%
subject = 'PetriceanuC'
cfg.init_current_subject(subject)
conditions = ['/'.join((ac, st)) for st in cfg.presented_stimuli for ac in cfg.action_classes]

# %%
epochs = mne.read_epochs(cfg.bipolar_ref_epochs_file)

# %%
sig_chs = ["P'7-P'8", "P'15-P'16", "G14-G15", "K'1-K'2", "J6-J7", "N10-N11"] # "N9-N10",
# sig_chs = ["P'15", "P'16", "G14", "G15", "K'1", "K'2", "J6", "J7", "N9", "N10", "N11"]
epochs = epochs.pick(sig_chs)

# %%
freqs = np.geomspace(5, 152, num=50)
n_cycles = freqs / freqs[0]
tfr_tmin, tfr_tmax = -0.3, 2.3

# %%
epochs_tfr= mne.time_frequency.tfr_morlet(
    epochs, freqs, n_cycles=n_cycles,
    average=False, return_itc=False, n_jobs=-3, verbose=False)
epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0), verbose=False)
epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)

# %%
avg_tfr_dict = {}
for cond in conditions:
    avg_tfr_dict[cond] = epochs_tfr[cond].average(method=lambda x : trim_mean(x, 0.1))

# %%
fig, axes = plt.subplots(
    nrows=len(conditions),
    ncols=len(sig_chs),
    sharex='all',
    sharey='all',
    figsize=(20, 10)

)

rows = conditions
cols = sig_chs

for row, ax in zip(rows, axes):
    avg_tfr_dict[row].plot(
        axes=ax, vmin=-1.5, vmax=1.5,  combine=None, show=False, colorbar=False)

for ax in axes.ravel():
    ax.set_xlabel(None)
    ax.set_ylabel(None)

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, size='large')

fig.tight_layout()
fig.show()
