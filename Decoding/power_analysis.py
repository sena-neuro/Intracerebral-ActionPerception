import numpy as np
import numpy.ma as ma
from mne.stats import permutation_cluster_test
import hdf_utils as hu
from collections import defaultdict

action_classes = hu.action_classes
hdf = hu.read_hdf()


def cluster_stat_on_power(region_list=hu.region_list()):
    nested_dict = lambda: defaultdict(nested_dict)
    T_obs = nested_dict()
    clusters = nested_dict()
    cluster_p_values = nested_dict()
    H_0 = nested_dict()

    for region in region_list:
        region_leads = hu.region_to_idx(region)
        arr_shape = (len(region_leads), 200, 50, 64)
        mean_power = {ac: np.empty(arr_shape, dtype=np.float32) for ac in action_classes}

        mask_arr = np.empty(arr_shape, dtype='|b1')
        power_arr = np.empty(arr_shape, dtype=np.float32)
        for ac in action_classes:
            z_dset = hdf[ac + '/z_power']
            z_dset.read_direct(power_arr, np.s_[region_leads, ...])

            mask_dset = hdf[ac + '/mask']
            mask_dset.read_direct(mask_arr, np.s_[region_leads, ...])

            mean_power[ac] = np.swapaxes(ma.mean(ma.masked_array(power_arr, mask=mask_arr), axis=0), 0, -1)

        for ac in action_classes:
            other_ac = filter(lambda x: x != ac, action_classes)
            mean_power_rest = np.concatenate((mean_power[next(other_ac)], mean_power[next(other_ac)]))

            T_obs[region][ac], clusters[region][ac], cluster_p_values[region][ac], H_0[region][ac] = \
                permutation_cluster_test([mean_power[ac], mean_power_rest], n_permutations=1024, tail=1, n_jobs=-1)
    return T_obs, clusters, cluster_p_values, H_0

# TODO Test
def filter_significant_clusters(p):
    if p.any() < 0.05:
        return True
    return False
