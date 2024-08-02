# %%
from scipy.stats import zmap, trimboth
import seeg_action.preprocessing as pp
from seeg_action.project_config import ProjectConfig as cfg
import mne
import numpy as np
import re

# %%
subject = 'PetriceanuC'
cfg.init_current_subject(subject)
epochs = mne.read_epochs(cfg.epochs_file)
epochs.pick_types(seeg=True)

# %%
bad_channels = ['P3']
epochs.drop_channels(bad_channels)

# %%
epochs = epochs.apply_function(lambda x: zmap(x, trimboth(x, 0.1)))

# %%
pattern = r'^([A-Z]\'?)\d+'
shafts = list(set(re.search(pattern, ch).groups()[0] for ch in epochs.ch_names))

# %%
epochs_bpr = epochs.copy()
for shaft in shafts:
    channels = [ch for ch in epochs.ch_names if re.search(pattern, ch).groups()[0] == shaft]
    epochs_bpr = mne.set_bipolar_reference(epochs_bpr,
                                           anode=channels[:-1],
                                           cathode=channels[1:])

 # %%
gm_channels = pp.get_gm_channels(epochs_bpr.info)
epochs_bpr.pick(gm_channels)