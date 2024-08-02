# %%
import matplotlib.pyplot as plt
import mne
from seeg_action import project_config as cfg
from seeg_action import preprocessing as pp
import numpy as np
from scipy.stats import trim_mean, zmap, trimboth

# %%
subject = 'FedericoR'
cfg.init_current_subject(subject)

# %%
# OP1
roi_chs = ["P'8", "Q'3", "Q'4", "Q'5", "Q'6", "Q'9", "W'7", "W'8", "W'9", "W'10"]

# %%
# let's explore some frequency bands
iter_freqs = [
    ('Theta', 4, 7),
    ('Mu', 7.5, 12.5),
    ('Alpha', 8, 12),
    ('SMR', 12.5, 15.5),
    ('Beta', 12, 30),
    ('Low-Gamma', 32, 100),
    ('High-Gamma', 80, 150)
]
conditions = ['/'.join((ac, st)) for st in cfg.presented_stimuli for ac in cfg.action_classes]

# %%
# (re)load the data to save memory
orig_raw = mne.io.read_raw_fif(cfg.filtered_raw_file)
orig_raw.pick(roi_chs)
# raw.pick_types(seeg=True, eog=True)
orig_raw.load_data()
frequency_map = {cond: list() for cond in conditions}
tmin, tmax, baseline = -0.5, 2.6, None
for band, fmin, fmax in iter_freqs:
    raw = orig_raw.copy()

    raw = pp.bandpass_filter(raw, l_freq=fmin, h_freq=fmax)

    # epoch
    all_events, event_id = mne.events_from_annotations(raw, cfg.event_description_to_id)
    epochs = mne.Epochs(raw, all_events, event_id=event_id,
                        preload=True,
                        tmin=-0.5, tmax=2.6,
                        baseline=baseline,
                        verbose=True
                        )

    epochs = epochs.apply_function(lambda x: zmap(x, trimboth(x, 0.1)))

    # remove evoked response
    epochs.subtract_evoked()

    # get analytic signal (envelope)
    epochs.apply_hilbert(envelope=True)
    epochs.crop(tmin=-0.2, tmax=2.)

    for cond in conditions:
        frequency_map[cond].append(
            ((band, fmin, fmax), epochs[cond].average(method=lambda x : trim_mean(x, 0.1, axis=0))))
        # epochs[cond].average()))

# %%
cond = 'BOD/ST'

# %%
# Helper function for plotting spread
def stat_fun(x):
    """Return sum of squares."""
    return np.sum(x ** 2, axis=0)

n_bands = len(iter_freqs)
# Plot
fig, axes = plt.subplots(n_bands, 1, figsize=(10, 7), sharex=True, sharey=True)
colors = plt.colormaps['winter_r'](np.linspace(0, 1, n_bands))
for ((freq_name, fmin, fmax), average), color, ax in zip(
        frequency_map[cond], colors, axes.ravel()[::-1]):
    times = average.times * 1e3
    gfp = np.sum(average.data ** 2, axis=0)
    gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
    ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
    ax.axhline(0, linestyle='--', color='grey', linewidth=2)
    ci_low, ci_up = mne.stats.bootstrap_confidence_interval(average.data, random_state=0,
                                                  stat_fun=stat_fun)
    ci_low = mne.baseline.rescale(ci_low, average.times, baseline=(None, 0))
    ci_up = mne.baseline.rescale(ci_up, average.times, baseline=(None, 0))
    ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
    ax.grid(True)
    ax.set_ylabel('GFP')
    ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                xy=(0.95, 0.8),
                horizontalalignment='right',
                xycoords='axes fraction')
    # ax.set_xlim(-.2, 2.)

axes.ravel()[-1].set_xlabel('Time [ms]')