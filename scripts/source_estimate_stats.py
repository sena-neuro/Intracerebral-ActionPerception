# %%
import mne
from seeg_action import project_config as cfg
import numpy as np
import h5py
import scipy.stats


cfg.init_config('PriviteraM')

tfr_path = cfg.derivatives_path / f'{cfg.current_subject}-tfr.h5'

if not tfr_path.exists():
    print('TFR file not found! Creating TFR file...')

    epochs = mne.read_epochs(cfg.epochs_file, preload=False)

    # define frequencies of interest
    freqs = np.geomspace(5, 152, num=50)
    n_cycles = freqs / freqs[0]

    with h5py.File(tfr_path, "w") as f:
        for cond in epochs.event_id:
            e = epochs[cond].load_data()
            e.pick_types(seeg=True)
            epochs_tfr = mne.time_frequency.tfr_morlet(e, freqs, n_cycles=n_cycles, zero_mean=False,
                                                       average=False, return_itc=False, n_jobs=-3, verbose='ERROR')

            epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0))
            epochs_tfr.crop(tmin=-0.3, tmax=2.4)

            f.create_dataset(cond, data=epochs_tfr.data)
            print("Time-frequency representation for {cond} is written.".format(
                cond=cond))
    print('[DONE] All conditions are stored in the TFR file!')

# %%
cond_order = []
with h5py.File(tfr_path, "r") as f:
    # Assuming n_trials same for each condition
    n_trials, n_channels, n_freqs, n_times = f[f'MN/ST'].shape

    data = np.empty((n_trials*9, n_channels, n_freqs, n_times), dtype=np.float32)
    start = 0
    for cond in cfg.event_code_to_id:
        dset = f[cond]
        end = start + n_trials
        dset.read_direct(data, dest_sel=np.s_[start:end, ...])
        start = end
        cond_order.extend([cond] * n_trials)


# %%
src = mne.read_source_spaces(cfg.oct_6_src_file)
adjacency = mne.spatial_src_adjacency(src)
# %%
factor_levels = [3, 3]
effects = "A"


def stat_fun(*args):
    return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                               effects=effects, return_pvals=False)[0]


# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001
n_replications = n_trials // 9

# ValueError: adjacency (len 8196) must be of the correct size, i.e. be equal to or evenly divide the number of tests (135050).
# If adjacency was computed for a source space, try using the fwd["src"] or inv["src"] as some original source space vertices can be excluded during forward computation

f_thresh = mne.stats.f_threshold_mway_rm(n_replications, factor_levels, effects, pthresh)
tail = 1  # f-test, so tail > 0
n_permutations = 1024
F_obs, clusters, cluster_p_values, _ = mne.stats.spatio_temporal_cluster_test(
    data, threshold=f_thresh, tail=tail, n_jobs=-2,
    check_disjoint=True, adjacency=adjacency, n_permutations=n_permutations,
    buffer_size=1000, out_type='indices')