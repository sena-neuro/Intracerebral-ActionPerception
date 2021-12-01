import h5py
from pathlib import Path
import numpy as np
import pickle
import numpy.ma as ma
from mne.stats import permutation_cluster_test
import hdf_utils

hdf = hdf_utils.read_hdf()
action_classes = ['MN', 'IP', 'SD']


def cluster_stat_on_power():
    region_list = hdf_utils.region_list()
    clusters = dict(fromkeys=region_list)
    cluster_p_values = dict(fromkeys=region_list)

    for region in region_list:
        region_leads = hdf_utils.region_to_idx(region)

        mean_power = {ac: np.empty((len(region_leads), 200, 50, 64), dtype=np.float32) for ac in action_classes}
        clusters[region] = dict(fromkeys=action_classes)
        cluster_p_values[region] = dict(fromkeys=action_classes)

        #     'MN': np.swapaxes(ma.mean(, axis=0), 0, -1),
        #     'IP': np.swapaxes(ma.mean(IP_power_dset[region_leads, :, :, :], axis=0), 0, -1),
        #     'SD': np.swapaxes(ma.mean(SD_power_dset[region_leads, :, :, :], axis=0), 0, -1)
        # }

        for ac in action_classes:
            dset = hdf[ac + '/z_power']
            dset.read_direct(mean_power[ac], np.s_[region_leads, ...])
            mean_power[ac] = np.swapaxes(ma.mean(mean_power[ac], axis=0), 0, -1)

        for ac in action_classes:
            other_ac = filter(lambda x: x != ac, action_classes)
            mean_power_rest = np.concatenate((mean_power[next(other_ac)], mean_power[next(other_ac)]))

            _, clusters[ac], cluster_p_values[ac], _ = \
                permutation_cluster_test([mean_power[ac], mean_power_rest], n_permutations=1024, tail=1, n_jobs=-1)

    return clusters, cluster_p_values
