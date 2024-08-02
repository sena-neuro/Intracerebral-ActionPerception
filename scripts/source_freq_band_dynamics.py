# %%
import matplotlib.pyplot as plt
import mne
from seeg_action import project_config as cfg
from seeg_action import preprocessing as pp
from scipy.stats import trim_mean, zmap, trimboth

# %%
subject = 'FedericoR'
cfg.init_current_subject(subject)

# %%
# let's explore some frequency bands
freq_band_map = {
#    'Theta': (4, 7),
#    'Mu': (7.5, 12.5),
#    'Alpha': (8, 12),
#    'SMR': (12.5, 15.5),
#    'Beta': (12, 30), # high end?
#    'Gamma': (21, 50),
#    'HFB': (50, 150)
    'Gamma': (80, 150)
}
conditions = ['/'.join((ac, st)) for st in cfg.presented_stimuli for ac in cfg.action_classes]

# %%
# (re)load the data to save memory
orig_raw = mne.io.read_raw_fif(cfg.filtered_raw_file)
# raw.pick_types(seeg=True, eog=True)
orig_raw.load_data()
orig_raw = orig_raw.apply_function(lambda x: zmap(x, trimboth(x, 0.1)))

# %%
power_map = {freq: {cond: None for cond in conditions} for freq in freq_band_map}

for band, (fmin, fmax) in freq_band_map.items():
    raw = orig_raw.copy()
    raw = pp.bandpass_filter(raw, l_freq=fmin, h_freq=fmax)

    # epoch
    all_events, event_id = mne.events_from_annotations(raw, cfg.event_description_to_id)
    epochs = mne.Epochs(raw, all_events, event_id=event_id,
                        preload=True,
                        tmin=-1., tmax=3.,
                        baseline=(-0.5, -0.1),
                        verbose=True
                        )
    epochs.apply_baseline()


    # get analytic signal (envelope)
    epochs.apply_hilbert(envelope=True)
    epochs.crop(tmin=-0.5, tmax=2.6)

    for cond in conditions:
        power_map[band][cond] =  epochs[cond].average(method=lambda x : trim_mean(x, 0.1, axis=0))

# %%
evk = power_map['Gamma']['OBJ/ST']
evk.apply_baseline((-0.4, -0.1))
evk.plot()
evk.plot_image()