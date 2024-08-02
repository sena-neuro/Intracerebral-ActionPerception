import mne
from seeg_action.project_config import ProjectConfig as cfg
import numpy as np

# https://github.com/mne-tools/mne-python/blob/maint/1.2/mne/source_estimate.py#L3207-L3417
def stcs_near_sources(epochs, distance=0.01, mode='weighted', surface='pial'):
    from scipy.spatial.distance import cdist, pdist
    from mne.source_estimate import (SourceEstimate, VolSourceEstimate,
                                     MixedSourceEstimate, read_surface)
    from mne.utils import _check_option

    src = mne.read_source_spaces(cfg.oct_6_src_file)

    # get channel positions that will be used to pinpoint where
    # in the Source space we will use the evoked data
    pos = epochs._get_channel_positions()

    # coord_frame transformation from native mne "head" to MRI coord_frame
    # convert head positions -> coord_frame MRI
    head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
    pos = mne.transforms.apply_trans(head_mri_t, pos)

    surf_rr = [read_surface(
        cfg.patients_path / cfg.current_subject / 'surf' / f'{hemi}.{surface}')[0] / 1000.
               for hemi in ('lh', 'rh')]
    rrs = np.concatenate([s_rr[s['vertno']] for s_rr, s in
                          zip(surf_rr, src)])

    # ensure it's a usable one
    klass = dict(
        surface=SourceEstimate,
        volume=VolSourceEstimate,
        mixed=MixedSourceEstimate,
    )
    _check_option('src.kind', src.kind, sorted(klass.keys()))
    klass = klass[src.kind]

    if src.kind == 'surface':
        pos = mne.surface._project_onto_surface(pos, dict(rr=rrs), project_rrs=True,
                                                method='nearest')[2]

    # compute pairwise distance between source space points and sensors
    dists = cdist(rrs, pos)
    assert dists.shape == (len(rrs), len(pos))

    # only consider vertices within our "epsilon-ball"
    # characterized by distance kwarg
    vertices = np.where((dists <= distance).any(-1))[0]
    w = np.maximum(1. - dists[vertices] / distance, 0)

    # now we triage based on mode
    if mode in ('single', 'nearest'):
        range_ = np.arange(w.shape[0])
        idx = np.argmax(w, axis=1)
        vals = w[range_, idx] if mode == 'single' else 1.
        w.fill(0)
        w[range_, idx] = vals
    elif mode == 'weighted':
        norms = w.sum(-1, keepdims=True)
        norms[norms == 0] = 1.
        w /= norms

    stcs = []
    for e in epochs:
        nz_data = w @ e
        data = np.zeros(
            (sum(len(s['vertno']) for s in src), len(epochs.times)),
            dtype=nz_data.dtype)
        data[vertices] = nz_data
        _vertices = [s['vertno'].copy() for s in src]

        stcs.append(klass(data, _vertices,
                          epochs.times[0], 1. / epochs.info['sfreq'],
                          subject=cfg.current_subject))
    return stcs
