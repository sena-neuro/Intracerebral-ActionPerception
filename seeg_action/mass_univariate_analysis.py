import numpy as np
import mne
import pandas as pd
from seeg_action import project_config as cfg
import statsmodels.api as sm


def mass_planned_contrast_anova(epochs):
    # Design matrix
    contrasts = [
        'action_vs_control',
        'static_vs_dynamic',
        'MN_vs_IP+SD',
        'IP_vs_SD']

    design = pd.DataFrame({'condition': np.vectorize(cfg.event_id_to_code.get)(epochs.events[..., -1])})
    design['action_vs_control'] = np.where(design.condition.str.contains('ST'), 2, -3)
    design['static_vs_dynamic'] = np.where(design.condition.str.contains('SC'), 1,  # Static control
                                           np.where(design.condition.str.contains('DC'), -1,  # Dynamic control
                                                    0))  # else action stimulus condition
    design['MN_vs_IP+SD'] = np.where(design.condition.str.contains('MN/ST'), 2,  # Manipulative actions
                                     np.where(
                                         (design.condition.str.contains('IP/ST') | design.condition.str.contains(
                                             'SD/ST')),
                                         -1,
                                         0))  # else control stimuli
    design['IP_vs_SD'] = np.where(design.condition.str.contains('IP/ST'), 1,  # Interpersonal actions
                                  np.where(design.condition.str.contains('SD/ST'), -1,  # Skin-displacing actions
                                           0))
    design.drop(columns='condition', inplace=True)
    design = sm.add_constant(design)

    results_map = {}
    for ch in epochs.ch_names:
        results_map[ch] = _channelwise_planned_contrast_anova(ch, epochs, design)

    return results_map


def _channelwise_planned_contrast_anova(channel, epochs, design):
    print("Currently, performing mass univariate analysis on channel {channel}".format(channel=channel))
    ch_epochs = epochs.copy().pick(channel)
    ch_epochs.crop(tmax=1.8)

    freqs = np.geomspace(5, 152, num=50)  # define frequencies of interest
    n_cycles = freqs / freqs[0]

    epochs_tfr = mne.time_frequency.tfr_morlet(ch_epochs, freqs, n_cycles=n_cycles,
                                               average=False, return_itc=False, n_jobs=-3)
    epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0))
    epochs_tfr.crop(tmin=-0.1, tmax=1.5)        # tmin=-0.3, tmax=2.3
    data = epochs_tfr.data

    power = mne.decoding.Vectorizer.transform(data)

    model = sm.OLS(endog=power, exog=design)
    results = model.fit()
    return results


