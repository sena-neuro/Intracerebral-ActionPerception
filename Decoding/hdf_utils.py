from logging import log
import h5py
from pathlib import Path
import numpy as np
import pickle
import get_leads_for_regions
import numpy.ma as ma

action_classes = ['MN', 'IP', 'SD']
if Path().owner() == 'senaer':
    parent_path = Path('/Users/senaer/Codes/CCNLab/Intracerebral-ActionPerception')
else:
    parent_path = Path('/auto/data2/oelmas/Intracerebral')
input_path = parent_path / 'Data' / 'TF_Analyzed'
output_path = parent_path / 'Data'
last_hdf_file = max(Path(output_path).rglob('*intracerebral_action_data.hdf5'),
                        key=lambda file: file.stat().st_mtime)


def read_hdf(mode='r'):
    return h5py.File(last_hdf_file, mode)


def __initialize_label_region_df():
    left_lead_file_path = output_path / "updated_left_lead_regions.csv"
    region_lead_labels_path = output_path / "updated_left_region_lead_labels.csv"

    # TODO add right regions

    # Dataframe in which one column is leads the other is regions that those leads are in
    return get_leads_for_regions.get_leads_for_regions(left_lead_file_path, region_lead_labels_path)

# TODO logs
def __init(reset=False):
    global lead_name_to_idx, label_region_df
    with h5py.File(last_hdf_file, 'r+') as hdf:
        lead_name_to_idx = pickle.loads(hdf.attrs['lead_name_to_idx'])
        label_region_df = __initialize_label_region_df()
        if reset:
            hdf_attrs = ['masked', 'z_scored']

            for a in hdf_attrs:
                try:
                    del hdf.attrs[a]
                except:
                    pass
            hdf_attrs = ['masked', 'z_scored', 'small_x', 'big_x']
            for ac in action_classes:
                ac_group = hdf[ac]
                for a in hdf_attrs:
                    try:
                        del ac_group.attrs[a]
                    except:
                        pass
                try:
                    del ac_group['mask']
                except:
                    pass
                try:
                    del ac_group['z_power']
                except:
                    pass
        __mask_power_dset(hdf)
        __add_z_scored_power_dset(hdf)
        # __add_region_dict(hdf)


def __mask_power_dset(hdf: h5py.File):
    if 'masked' in hdf.attrs and hdf.attrs['masked']:
        log('Raw power data is already masked!')
    else:
        for ac in action_classes:
            ac_group = hdf[ac]
            power_arr = np.empty(ac_group['raw_power'].shape, dtype=np.float32)
            ac_group['raw_power'].read_direct(power_arr)
            ma_power_arr = ma.masked_invalid(power_arr)
            ma_power_arr = ma.masked_outside(ma_power_arr, 0, 1000)

            ac_group.create_dataset('mask', data=ma_power_arr.mask)
            ac_group.attrs['masked'] = True
        hdf.attrs['masked'] = True


def __add_z_scored_power_dset(hdf : h5py.File):
    if 'z_scored' in hdf.attrs and hdf.attrs['z_scored']:
        log('HDF already includes z-scored power!')
    else:
        for ac in action_classes:
            ac_group = hdf[ac]

            raw_power_dset = ac_group['raw_power']
            power_arr = np.empty(raw_power_dset.shape, dtype=np.float32)
            raw_power_dset.read_direct(power_arr)

            mask_dset = ac_group['mask']
            mask_arr = np.empty(mask_dset.shape, dtype='|b1')
            mask_dset.read_direct(mask_arr)

            ac_group.create_dataset('z_power', data=__z_score(ma.masked_array(power_arr, mask=mask_arr)))
            ac_group.attrs['z_scored'] = True
        hdf.attrs['z_scored'] = True


def __z_score(power):
    std_baseline = ma.std(power[:, 0:7, :, :], axis=1, keepdims=True)
    mean_baseline = ma.mean(power[:, 0:7, :, :], axis=1, keepdims=True)
    return (power - mean_baseline) / std_baseline


def get_all_regions():
    return list(label_region_df.region_idx.unique())


# Prob no need bcs we already have the region_to_idx() method
def __add_region_dict(hdf: h5py.File):
    if 'region_to_leads' in hdf.attrs:
        log('HDF already includes the region_to_leads dictionary!')
    else:
        all_region_names = get_all_regions()
        region_to_lead_idx = dict(zip(all_region_names, map(region_to_idx, all_region_names)))
        hdf.attrs['region_to_lead_idx'] = pickle.dumps(region_to_lead_idx, protocol=0)


def region_list():
    if 'label_region_df' not in globals():
        global label_region_df
        label_region_df = __initialize_label_region_df()
    return list(label_region_df.region_idx.unique())


def region_to_idx(region):
    if 'label_region_df' not in globals():
        global label_region_df
        label_region_df = __initialize_label_region_df()
    if 'lead_name_to_idx' not in globals():
        global lead_name_to_idx
        with read_hdf() as hdf:
            lead_name_to_idx = pickle.loads(hdf.attrs['lead_name_to_idx'])
    # TODO for later, add multiple regions option
    lead_names_in_region = label_region_df[label_region_df.region_idx == region].lead_idx.values
    lead_idx_in_region = list(
        map(lambda s: lead_name_to_idx[s], filter(lambda s: s in lead_name_to_idx, lead_names_in_region)))
    lead_idx_in_region.sort()
    return lead_idx_in_region


def __leads_by_region():
    if 'label_region_df' not in globals():
        global label_region_df
        label_region_df = __initialize_label_region_df()
    n_leads_by_region = label_region_df.groupby('region_idx').count()
    print(n_leads_by_region)


def idx_to_leadname(lead_idx):
    if 'lead_name_to_idx' not in globals():
        global lead_name_to_idx
        with read_hdf() as hdf:
            lead_name_to_idx = pickle.loads(hdf.attrs['lead_name_to_idx'])
    return list(lead_name_to_idx.keys())[lead_idx]