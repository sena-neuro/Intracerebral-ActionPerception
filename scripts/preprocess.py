# %%
import mne
from seeg_action import project_config as cfg
from seeg_action import preprocessing as pp
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt


# parser = cfg.init_argparse()
# args = parser.parse_args()
# cfg.init_config(args.subject_name)
cfg.init_config('PriviteraM')

# %%
# STEP 1: import .EDF or . EEG files, add events, export .fif
if not cfg.raw_fif_save_file.exists():
    pp.export_raw_fif()

# %% STEP 2: Filtering: low pass at 1.5 hz high pass at 300 hz notch filter at 50 hz and its harmonics
if not cfg.filtered_raw_file.exists():
    pp.export_filtered_raw_fif()

# %% STEP 3: Artifact rejection
if not cfg.ica_file.exists():
    pp.export_ica_solution()

# read ica sol
#ica = mne.preprocessing.read_ica(cfg.ica_file)
#ica.apply(epochs)

# %%
raw = mne.io.read_raw_fif(cfg.filtered_raw_file).load_data()
ica = mne.preprocessing.read_ica(cfg.ica_file)
ica.plot_sources(raw, block=True)
ica.plot_overlay(raw, exclude=ica.exclude)
ica.apply(raw)

# %%
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine='mean', evoked=True)

# %% STEP 4: Epoching
if not cfg.epochs_file.exists():
    pp.export_epochs()

# %%
if not cfg.epochs_action_file.exists():
    pp.export_action_minus_control_epochs()

# %%
epochs = mne.read_epochs(cfg.epochs_file)
epochs.pick_types(seeg=True)

# %%

orig_epochs = epochs.copy()
# %%
epochs.crop(tmin=-0.2, tmax=1.)
evk = {cond: epochs[cond].average() for cond in epochs.event_id}
# mne.viz.plot_compare_evokeds(evk, colors={'MN': 'red', 'IP': 'blue', 'SD': 'green'},
#                             linestyles={'ST': 'solid', 'SC': ':', 'DC': '--'})
# %%
info = epochs.info
electrodes = list(set(ch[0] for ch in epochs.ch_names))


def my_callback(ax, ch_idx):
    """
    This block of code is executed once you click on one of the channel axes
    in the plot. To work with the viz internals, this function should only take
    two parameters, the axis and the channel or data index.
    """
    mne.viz.plot_compare_evokeds(evk,
                                 picks=[ch_idx],
                                 colors={'OBJ': 'red', 'PER': 'blue', 'BOD': 'green'},
                                 linestyles={'ST': 'solid', 'CS': ':', 'CD': '--'},
                                 axes=ax,
                                 show=False,
                                 legend=True,
                                 split_legend=True,
                                 show_sensors=False)


for e in electrodes:
    fig = plt.figure(e)
    e_picks = mne.pick_channels_regexp(epochs.ch_names, e)
    print([info.ch_names[i] for i in e_picks])
    layout = mne.channels.make_grid_layout(info, picks=e_picks, n_col=5)
    for ax, idx in mne.viz.iter_topography(info,
                                           layout=layout,
                                           fig=fig,
                                           fig_facecolor='white', axis_facecolor='white', axis_spinecolor='white',
                                           on_pick=my_callback,
                                           legend=False,
                                           layout_scale=0.8):
        mne.viz.plot_compare_evokeds(evk,
                                     picks=[idx],
                                     colors={'OBJ': 'red', 'PER': 'blue', 'BOD': 'green'},
                                     linestyles={'ST': 'solid', 'CS': ':', 'CD': '--'},
                                     axes=ax,
                                     show=False,
                                     legend=False,
                                     show_sensors=False)
    fig.suptitle(f'Evoked responses in {e}')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
plt.show()

# STEP 5: Regression
if cfg.epochs_file.exists():
    raw = mne.read_epochs(cfg.epochs_file)