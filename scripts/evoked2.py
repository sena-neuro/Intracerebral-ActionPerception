# %%
import pandas as pd
from seeg_action.project_config import ProjectConfig as cfg
import mne
import numpy as np
from scipy.stats import zmap, trimboth, trim_mean
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
action_induced_epc = epochs['ST']
for cond in action_induced_epc.event_id:
    action_class, _, action_examplar, object_size, actor = cond.split('/')
    evk_cs = epochs['/'.join([action_class, 'CS', action_examplar, object_size, actor])].average()
    evk_cd = epochs['/'.join([action_class, 'CD', action_examplar, object_size, actor])].average()
    evk_control = mne.combine_evoked([evk_cs, evk_cd], [0.5, 0.5])
    action_induced_epc[cond].subtract_evoked(evk_control)

# %%
evk_dict = {}
for cond in conditions: # cfg.action_classes
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
    picks=['G13-G14'],
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
