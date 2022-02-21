import mne
import numpy as np
from pathlib import Path

raw_data_path = Path('/auto/data/burgen/StereoEEG_ActionBase')
eeg_file = raw_data_path / 'EEGdata' / 'BerberiM_ActionBase.EEG'


raw = mne.io.read_raw_nihon(eeg_file)


seeg_picks = mne.pick_channels_regexp(raw.ch_names, regexp='^[A-Z]\'(\d+)')
# eeg_picks = mne.pick_channels_regexp(raw.ch_names, regexp='^[A-Z](\d+)')
eog_picks = mne.pick_channels_regexp(raw.ch_names, "EOG")
stim_picks = mne.pick_channels_regexp(raw.ch_names, "DC")

channel_type_dict = dict()
for idx in seeg_picks:
    channel_type_dict[raw.ch_names[idx]] = 'seeg'

for idx in eog_picks:
    channel_type_dict[raw.ch_names[idx]] = 'eog'

for idx in stim_picks:
    channel_type_dict[raw.ch_names[idx]] = 'stim'

raw.set_channel_types(channel_type_dict)
raw.pick_types(seeg=True, stim=True, eog=True)

events = mne.find_events(raw, min_duration=0.001, initial_event=False, consecutive=False)
events = events[:-1]

event_log_file = raw_data_path / 'LOG' / 'ActionBase_berberiM_detailed.txt'
event_codes = np.loadtxt(event_log_file, dtype='i4', usecols=3)
events[:, 2] = event_codes

event_dict = {
    1: 'IP-ST',
    2: 'IP-DC',
    3: 'IP-SC',
    4: 'MN-ST',
    5: 'MN-DC',
    6: 'MN-SC',
    7: 'SD-ST',
    8: 'SD-DC',
    9: 'SD-SC'}

annot_from_events = mne.annotations_from_events(
    events=events, event_desc=event_dict, sfreq=raw.info['sfreq'],
    orig_time=raw.info['meas_date'])
raw.set_annotations(annot_from_events)

inv_event_dict = {v: k for k, v in event_dict.items()}
all_events, all_event_id = mne.events_from_annotations(raw, event_id=inv_event_dict)
#
# # Trial length is actually -0.5 , 2.6
# picks = mne.pick_types(raw.info, seeg=True, eeg=False, eog=False, stim=False)
# epochs = mne.Epochs(raw, all_events, picks=picks, tmin=-0.5, tmax=2.6, baseline=(-0.5, 0))
#                     # ,reject={'seeg':1000})

b_picks = mne.pick_channels_regexp(raw.ch_names, regexp='^B\'')
raw.copy().crop(tmin=100, tmax=200).plot(order=b_picks, scalings='auto')
# epochs.plot(n_epochs=10, picks=b_picks, scalings='auto')

# freqs = np.arange(50, 160, 10)
# n_cycles = freqs * .1
# power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, use_fft=True, n_jobs=4)
#
#
# def plot_gamma_for_itc_detection(lead_no, mode='logratio'):
#     power.plot([lead_no], baseline=(-0.5, 0), mode=mode, title=power.ch_names[lead_no])

# Looked weird
#
#
# raw.copy().pick_types(seeg=True).plot(duration=60, scalings='auto')
#
#
# def add_arrows(axes):
#     for ax in axes:
#         freqs = ax.lines[-1].get_xdata()
#         psds = ax.lines[-1].get_ydata()
#         for freq in (50, 100, 150):
#             idx = np.searchsorted(freqs, freq)
#             # get ymax of a small region around the freq. of interest
#             y = psds[(idx - 4):(idx + 5)].max()
#             ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
#                      width=0.1, head_width=3, length_includes_head=True)
#
#
# fig = raw.plot_psd(fmax=155, average=True)
# add_arrows(fig.axes[:2])
#
# seeg_picks = mne.pick_types(raw.info, seeg=True)
# freqs = np.arange(50, 251, 50)
# raw_notch_fir = raw.copy().notch_filter(freqs=freqs, picks=seeg_picks, trans_bandwidth=0.04)
# for title, data in zip(['Un', 'Notch (fir)'], [raw, raw_notch_fir]):
#     fig = data.plot_psd(fmax=155, average=True)
#     fig.subplots_adjust(top=0.85)
#     fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
#     add_arrows(fig.axes[:2])
#
# # raw_notch_iir = raw.copy().notch_filter(
# #     freqs=freqs, picks=seeg_picks, method='iir',
# #     iir_params=dict(order=6, ftype='butter'))
# # for title, data in zip(['Un', 'Notch (iir)'], [raw, raw_notch_iir]):
# #     fig = data.plot_psd(fmax=155, average=True)
# #     fig.subplots_adjust(top=0.85)
# #     fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
# #     add_arrows(fig.axes[:2])
#

#
