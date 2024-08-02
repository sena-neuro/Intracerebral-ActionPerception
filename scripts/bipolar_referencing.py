# %%
from seeg_action.project_config import ProjectConfig as cfg
import mne
import re

# %%
subject = 'PetriceanuC'
cfg.init_current_subject(subject)
conditions = ['/'.join((ac, st)) for ac in cfg.action_classes for st in cfg.presented_stimuli]

# %%
epochs = mne.read_epochs(cfg.clean_epochs_file)

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
epochs_bpr.save(cfg.bipolar_ref_epochs_file, overwrite=True)