# %%
import mne
from seeg_action import project_config as cfg
from seeg_action import preprocessing as pp
from scipy.stats import trim_mean, zmap, trimboth
import numpy as np

# %%
subject = 'FedericoR'
cfg.init_current_subject(subject)

# %%
conditions = ['/'.join((ac, st)) for st in cfg.presented_stimuli for ac in cfg.action_classes]
freqs = np.linspace(50, 150, 11)
n_cycles = freqs / freqs[0]
tfr_tmin, tfr_tmax = -0.5, 2.6
baseline = (-0.5, -0.05)

# %%
raw = mne.io.read_raw_fif(cfg.filtered_raw_file)
raw.drop_channels(["U'3", "E'6", "E'7", "P'10", "P'8", "C'17",
                   "N'1", "N'2", "P'1", "P'2", "E'1", "E'2", "F'1", "F'2"])
# raw.pick_types(seeg=True, eog=True)
raw.load_data()

# %%
# Should I scale each channel? -> ?
raw = raw.apply_function(lambda x: zmap(x, trimboth(x, 0.1)))

# %%
# epoch
all_events, event_id = mne.events_from_annotations(raw, cfg.event_description_to_id)
epochs = mne.Epochs(raw, all_events, event_id=event_id,
                    preload=True,
                    tmin=-1., tmax=3.,
                    reject=dict(seeg=3e1),
                    flat=dict(seeg=1e-1),
                    reject_tmin=-0.5,
                    reject_tmax=2.6,
                    baseline=baseline,
                    verbose=True
                    )
del raw

# %%
power = dict()
for cond in conditions:
    _epochs_tfr = mne.time_frequency.tfr_morlet(
        epochs[cond], freqs, n_cycles=n_cycles,
        average=False, return_itc=False, n_jobs=-3, verbose=False)

    _epochs_tfr.apply_baseline(baseline=baseline, mode='zscore')
    _epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)

    _epochs_tfr = _epochs_tfr.average(dim='freqs')
    power[cond] = _epochs_tfr.average(dim='epochs')


