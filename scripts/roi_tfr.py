# %%
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from itertools import combinations

import mne
from mne.stats import permutation_cluster_1samp_test

from seeg_action import project_config as cfg
import numpy as np
from scipy.stats import trim_mean, t, zmap, trimboth, trim1, tmean, sigmaclip

# %%
subject = 'FedericoR'
cfg.init_current_subject(subject)

# %%
epochs = mne.read_epochs(cfg.epochs_file)
epochs.pick_types(seeg=True)

# %%
# OP1
roi_chs = ["P'8", "Q'3", "Q'4", "Q'5", "Q'6", "Q'9", "W'7", "W'8", "W'9", "W'10"]
epochs.pick(roi_chs)

# %%
channels = roi_chs

import re
import numpy as np
from scipy.sparse import csr_matrix

# Function to get the neighbors of a channel
def get_neighbors(channel, channels):
    # Extract the shaft and electrode number
    shaft, electrode = re.match(r"([A-Z])'(\d+)", channel).groups()
    electrode = int(electrode)

    # Get the indices of the neighboring channels
    neighbor_indices = [channels.index(channel)]
    for neighbor_electrode in [electrode - 1, electrode + 1]:
        neighbor = f"{shaft}'{neighbor_electrode}"
        if neighbor in channels:
            neighbor_indices.append(channels.index(neighbor))
    return neighbor_indices

# Compute the neighbors for each channel
neighbors = [get_neighbors(channel, channels) for channel in channels]

# Create the csr_matrix
rows = np.repeat(np.arange(len(channels)), [len(row) for row in neighbors])
cols = np.hstack(neighbors)
data = np.ones(len(cols), dtype=int)

ch_adjacency = csr_matrix((data, (rows, cols)), shape=(len(channels), len(channels)))


# %%
freqs = np.geomspace(5, 152, num=50)
n_cycles = freqs / freqs[0]
tfr_tmin, tfr_tmax = -0.3, 2.3

# %%
conditions = ['/'.join((ac, st)) for st in cfg.presented_stimuli for ac in cfg.action_classes]

# %%
# Statistics
# We want a two-tailed test
tail = 0
alpha = 0.001

# Set the number of permutations to run.
n_permutations = 1000

# %%
avg_tfr_dict = {}
sig_mask = {}
for cond in conditions:
    sig_mask[cond] = {}

    epochs_tfr = mne.time_frequency.tfr_morlet(
        epochs[cond], freqs, n_cycles=n_cycles,
        average=False, return_itc=False, n_jobs=-3, verbose=False)

    epochs_tfr.apply_baseline(baseline=(tfr_tmin, tfr_tmax), mode='zscore')
    epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)

    for ch in roi_chs:
        _roi_power = epochs_tfr.copy().pick(ch).data

        # Decide
        # Alternative: trim1(roi_power, 0.05, tail='right', axis=0)
        _roi_power = np.clip(_roi_power, -5, 5).squeeze()

        n_trials, n_freqs, n_times = _roi_power.shape

        # Because we conduct a two-tailed test, we divide the p-value by 2 (which means
        # we're making use of both tails of the distribution).
        # As the degrees of freedom, we specify the number of observations (-1)
        # Finally, we subtract 0.001 / 2 from 1, to get the critical t-value
        # on the right tail (this is needed for MNE-Python internals)
        degrees_of_freedom = n_trials - 1
        t_thresh = t.ppf(1 - alpha / 2, df=degrees_of_freedom)

        adjacency = mne.stats.combine_adjacency(
            n_freqs,
            n_times
        )

        # Run the analysis
        # clu_stats[cond] =
        T_obs, clusters, cluster_p_values, H0 = \
            mne.stats.permutation_cluster_1samp_test(
                _roi_power, n_permutations=n_permutations,
                threshold=t_thresh, tail=tail,
                out_type='mask',
                adjacency=adjacency,
                verbose=True,
                n_jobs=-3
        )

        # Compute sig mask
        _idx = np.array(cluster_p_values < 0.05)
        _sig_clusters = np.array(clusters)[_idx]

        sig_mask[cond][ch] = np.any(_sig_clusters, axis=0)
    avg_tfr_dict[cond] = epochs_tfr.average(method=lambda x: tmean(x, limits=(-5, 5), axis=0))

# %%
cols = conditions
rows = roi_chs
# Create a GridSpec with an additional column for the colorbar
gs = GridSpec(len(rows), len(cols) + 1, width_ratios=[1] * len(cols) + [0.05])

vmin=-3
vmax=3
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = 'RdBu_r'
sm = cm.ScalarMappable(cnorm, cmap=cmap)

fig = plt.figure(figsize=(16, 10))

# Define a function to create axes based on the GridSpec
def create_axes(row, col):
    return plt.subplot(gs[row, col])

kwargs = dict(
    baseline=(-0.3, -0.1),
    mode='zscore',
    tmin=tfr_tmin,
    tmax=tfr_tmax,
    vmin=vmin,
    vmax=vmax,
    colorbar=False,
    combine=None,
    show=False,
    cnorm=cnorm,
    cmap=cmap,
    title=None,
    #mask_style='both',
    #mask_alpha=0.2,
)

for c, col in enumerate(cols):
    for r, row in enumerate(rows):
        ax = create_axes(r, c)
        ax.set_title(col if r == 0 else '')

        avg_tfr_dict[col].plot(
            picks=row,
            axes=ax,
        #    mask=sig_mask[col][row],
            **kwargs
        )
        if c > 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        else:
            ax.set_ylabel(row, rotation=10, size='x-large')

# Create a separate axis for the colorbar
cbar_ax = create_axes(0, len(cols))

# Compute the position and size of the colorbar
x0 = cbar_ax.get_position().x0
y0 = 0.15
width = 0.02
    # cbar_ax.get_position().width * 2  # Increase the width by a factor of 2
height = 0.7

# Set the position and size of the colorbar
cbar_ax.set_position(Bbox.from_bounds(x0, y0, width, height))
fig.colorbar(sm, cax=cbar_ax)

fig.show()

# %%
clu_stats = {}
for ac in cfg.action_classes:
    _roi_power = epochs_tfr[ac].data
    _roi_power = trim1(_roi_power, 0.05, tail='right', axis=0)
    _roi_power = _roi_power.reshape((-1, _roi_power.shape[2], _roi_power.shape[3]))


    # We want a two-tailed test
    tail = 0
    alpha = 0.001


    # Because we conduct a two-tailed test, we divide the p-value by 2 (which means
    # we're making use of both tails of the distribution).
    # As the degrees of freedom, we specify the number of observations (-1)
    # Finally, we subtract 0.001 / 2 from 1, to get the critical t-value
    # on the right tail (this is needed for MNE-Python internals)
    degrees_of_freedom = _roi_power.shape[0] - 1
    t_thresh = t.ppf(1 - alpha / 2, df=degrees_of_freedom)

    # Set the number of permutations to run.
    n_permutations = 5000

    # Run the analysis
    clu_stats[ac] = \
        permutation_cluster_1samp_test(_roi_power, n_permutations=n_permutations,
                                       threshold=t_thresh, tail=tail,
                                       out_type='mask',
                                       verbose=True,
                                       n_jobs=-3)

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for col, ax in zip(cfg.action_classes, axes):
    T_obs, clusters, cluster_p_values, H0 = clu_stats[col]
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]


    vmax = np.max(np.abs(T_obs))
    vmin = -vmax
    ax.imshow(T_obs, cmap=plt.cm.gray,
               extent=[tfr_tmin, tfr_tmax, freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    im = ax.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
               extent=[tfr_tmin, tfr_tmax, freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(col)
    fig.colorbar(im, ax=ax)
fig.suptitle(f'Induced power')
fig.tight_layout()
fig.show()



# %% OvO actions t-test
################################################################################
clu_stats_pairs = {}
pairs = list(combinations(cfg.action_classes, 2))
for ac_pair in pairs:

    _roi_power = []
    for ac in ac_pair:
        _power = epochs_tfr[ac].data
        _power = trim1(_power, 0.05, axis=0)
        _power = _power.reshape((-1, _power.shape[2], _power.shape[3]))
        _roi_power.append(_power)



    # We want a two-tailed test
    tail = 0
    alpha = 0.001


    # Because we conduct a two-tailed test, we divide the p-value by 2 (which means
    # we're making use of both tails of the distribution).
    # As the degrees of freedom, we specify the number of observations (-1)
    # Finally, we subtract 0.001 / 2 from 1, to get the critical t-value
    # on the right tail (this is needed for MNE-Python internals)
    degrees_of_freedom = _roi_power[0].shape[0] - 1
    t_thresh = t.ppf(1 - alpha / 2, df=degrees_of_freedom)

    # Set the number of permutations to run.
    n_permutations = 1000

    # Run the analysis
    clu_stats_pairs[ac_pair] = \
        mne.stats.permutation_cluster_test(
            _roi_power,
            n_permutations=n_permutations,
            threshold=t_thresh, tail=tail,
            out_type='mask',
            verbose=True,
            n_jobs=-3)

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for col, ax in zip(pairs, axes):
    T_obs, clusters, cluster_p_values, H0 = clu_stats_pairs[col]
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]


    vmax = np.max(np.abs(T_obs))
    vmin = -vmax
    ax.imshow(T_obs, cmap=plt.cm.gray,
               extent=[tfr_tmin, tfr_tmax, freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    im = ax.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
               extent=[tfr_tmin, tfr_tmax, freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(col)
    fig.colorbar(im, ax=ax)
fig.suptitle(f'Induced power')
fig.tight_layout()
fig.show()

# %%
################################################################################
################################################################################
################################################################################
cs_clu_stats = {}
for ac in cfg.action_classes:


    st_power = epochs_tfr[ac + '/ST'].data
    st_power = trim1(st_power, 0.05, axis=0)
    st_power = st_power.reshape((-1, st_power.shape[2], st_power.shape[3]))

    cs_power = epochs_tfr[ac + '/CS'].data
    cs_power = trim1(cs_power, 0.05, axis=0)
    cs_power = cs_power.reshape((-1, cs_power.shape[2], cs_power.shape[3]))

    # We want a two-tailed test
    tail = 0
    alpha = 0.001


    # Because we conduct a two-tailed test, we divide the p-value by 2 (which means
    # we're making use of both tails of the distribution).
    # As the degrees of freedom, we specify the number of observations (-1)
    # Finally, we subtract 0.001 / 2 from 1, to get the critical t-value
    # on the right tail (this is needed for MNE-Python internals)
    degrees_of_freedom = st_power[0].shape[0] - 1
    t_thresh = t.ppf(1 - alpha / 2, df=degrees_of_freedom)

    # Set the number of permutations to run.
    n_permutations = 1000

    # Run the analysis
    cs_clu_stats[ac] = \
        mne.stats.permutation_cluster_test(
            [st_power, cs_power],
            n_permutations=n_permutations,
            threshold=t_thresh, tail=tail,
            out_type='mask',
            verbose=True,
            n_jobs=-3)

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for col, ax in zip(cfg.action_classes, axes):
    T_obs, clusters, cluster_p_values, H0 = cs_clu_stats[col]
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]


    vmax = np.max(np.abs(T_obs))
    vmin = -vmax
    ax.imshow(T_obs, cmap=plt.cm.gray,
               extent=[tfr_tmin, tfr_tmax, freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    im = ax.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
               extent=[tfr_tmin, tfr_tmax, freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(col)
    fig.colorbar(im, ax=ax)
fig.suptitle(f'Induced power')
fig.tight_layout()
fig.show()


# %%
conditions = ['/'.join((ac, st)) for st in cfg.presented_stimuli for ac in cfg.action_classes]
avg_tfr_dict = {}
for cond in conditions:
    epochs_tfr = mne.time_frequency.tfr_morlet(
        epochs[cond], freqs, n_cycles=n_cycles,
        average=False, return_itc=False, n_jobs=-3, verbose=False)

    avg_tfr = epochs_tfr.average(method=lambda x : tmean(x, limits=(-50, 50), axis=0))
    avg_tfr.apply_baseline(mode='zscore', baseline=(-0.3, -0.1), verbose=False)
    avg_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)
    avg_tfr_dict[cond] = avg_tfr

# %%
fig, axes = plt.subplots(nrows=1, ncols=3)

kwargs = dict(
    combine='rms',
    show=False,
    colorbar=True,
)

for col, ax in zip(cfg.action_classes, axes):
    avg_tfr_dict[col].plot(
        axes=ax,
        **kwargs
    )
    ax.set_title(col)

fig.tight_layout(w_pad=0.)
fig.show()

# %%
cols = conditions
rows = roi_chs

fig, axes = plt.subplots(
    nrows=len(rows), ncols=len(cols),
    sharex='all',
    sharey='all',
)

kwargs = dict(
    vmin=-2.5,
    vmax=2.5,
    combine=None,
    show=False,
    colorbar=True,
)

for col, ax in zip(cols, zip(*axes)):
    avg_tfr_dict[col].plot(
        axes=list(ax),
        **kwargs
    )

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=10, size='x-large')

fig.show()
