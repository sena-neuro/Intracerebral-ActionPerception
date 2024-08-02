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
spectrum = raw.compute_psd(fmin=50., fmax=300., n_jobs=-3)
spectrum.plot()

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
epochs.load_data()
_epochs_tfr = mne.time_frequency.tfr_morlet(
    epochs, freqs, n_cycles=n_cycles,
    average=False, return_itc=False, n_jobs=-3, verbose=False)
del epochs

_epochs_tfr = _epochs_tfr.apply_baseline(baseline=baseline, mode='zscore')
_epochs_tfr = _epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)

# %%
mean_power = _epochs_tfr.average(dim='freqs')
del _epochs_tfr

# %%
e = mne.EpochsArray(
    data=mean_power.data.squeeze(),
    info=mean_power.info,
    events=mean_power.events,
    tmin=mean_power.tmin,
    event_id=mean_power.event_id,
    baseline=mean_power.baseline,
)
orig_e = e.copy()

# %%
# spectrum = e.compute_psd(fmin=50., fmax=150., n_jobs=-3)

# %% maybe for finding bad channels
# e.copy().drop_bad(reject=dict(seeg=5e1))

# %%
e.drop_bad(reject=dict(seeg=1e4))
e.copy().drop_bad(reject=dict(seeg=5e2))
# e.copy().drop_bad(reject=dict(seeg=5e2))
#     Rejecting  epoch based on SEEG : ["P'2"]
#     Rejecting  epoch based on SEEG : ["P'2"]
#     Rejecting  epoch based on SEEG : ["X'7"]
#     Rejecting  epoch based on SEEG : ["B'1", "B'6", "B'9", "B'10", "B'11", "B'13", "B'14", "R'1"]
#     Rejecting  epoch based on SEEG : ["P'3"]
#     Rejecting  epoch based on SEEG : ["P'3"]
#     Rejecting  epoch based on SEEG : ["P'3"]
#     Rejecting  epoch based on SEEG : ["Q'3"]
#     Rejecting  epoch based on SEEG : ["P'3", "P'4"]
#     Rejecting  epoch based on SEEG : ["P'2"]
#     Rejecting  epoch based on SEEG : ["P'3"]
#     Rejecting  epoch based on SEEG : ["P'2"]
# 12 bad epochs dropped

# %%
order=np.argsort(e.events[:, -1])
# No combine vmin=-1e3, vmax=1e3,
# GFP vmin=0, vmax=1e4,
e.plot_image(
    vmin=0,
    vmax=3e4,
    order=order,
    #scalings=dict(seeg=1e4),
    #units=dict(seeg='z-score'),
)

# %%
power = dict()
for cond in conditions:
    power[cond] = e[cond].average()

# %%