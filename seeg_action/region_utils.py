import mne
from seeg_action.project_config import ProjectConfig as cfg
from nilearn import datasets

def find_region_names_using_cut_coords(coords, atlas_img, labels=None):
    import numpy as np
    from nilearn.image.resampling import coord_transform
    from nilearn._utils.niimg import _safe_get_data
    """Given list of MNI space coordinates, get names of the brain regions.
    Names of the brain regions are returned by getting nearest coordinates
    in the given `atlas_img` space iterated over the provided list of
    `coords`. These new image coordinates are then used to grab the label
    number (int) and name assigned to it. Last, these names are returned.
    Parameters
    ----------
    coords : Tuples of coordinates in a list
        MNI coordinates.
    atlas_img : Nifti-like image
        Path to or Nifti-like object. The labels (integers) ordered in
        this image should be sequential. Example: [0, 1, 2, 3, 4] but not
        [0, 5, 6, 7]. Helps in returning correct names without errors.
    labels : str in a list
        Names of the brain regions assigned to each label in atlas_img.
        NOTE: label with index 0 is assumed as background. Example:
            harvard oxford atlas. Hence be removed.
    Returns
    -------
    new_labels : int in a list
        Labels in integers generated according to correspondence with
        given atlas image and provided coordinates.
    names : str in a list
        Names of the brain regions generated according to given inputs.
    """


    affine = atlas_img.affine
    atlas_data = _safe_get_data(atlas_img, ensure_finite=True)
    check_labels_from_atlas = np.unique(atlas_data)



    coords = list(coords)
    nearest_coordinates = []

    for sx, sy, sz in coords:
        nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        nearest_coordinates.append(nearest)

    assert(len(nearest_coordinates) == len(coords))

    new_labels = []
    names = []
    for coord_ in nearest_coordinates:
        # Grab index of current coordinate
        index = atlas_data[coord_]
        new_labels.append(index)
        if labels is not None:
            names.append(labels[index])

    return new_labels, names

