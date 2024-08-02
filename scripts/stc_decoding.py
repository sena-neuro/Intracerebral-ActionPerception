# %%
import mne
from seeg_action.project_config import ProjectConfig as cfg
from seeg_action import preprocessing as pp
from seeg_action.source_utils import stcs_near_sources
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# %%
cfg.init_current_subject('PriviteraM')
mne.set_log_file('ERROR')

epochs = mne.read_epochs(cfg.epochs_file)
epochs.pick_types(seeg=True)
epochs.drop_channels(pp.get_wm_channels(epochs.info))

# %%
stcs = stcs_near_sources(epochs)

# %%
# Retrieve source space data into an array
X = np.array([stc.data for stc in stcs])
y = np.vectorize(cfg.event_id_to_description.get)(epochs.events[..., -1]).astype('U6')

# %%
clf = make_pipeline(
    mne.decoding.Vectorizer(),
    LinearDiscriminantAnalysis()
)

# %%
time_decod = mne.decoding.SlidingEstimator(
    clf, n_jobs=-3, verbose=False)

# %%
# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y)

# %%

# Retrieve patterns after inversing the z-score normalization step:
patterns = mne.decoding.get_coef(time_decod, 'patterns_', inverse_transform=True)

# TODO Check if all stc's has the same vertno
stc = stcs[0]  # for convenience, lookup parameters from first stc
# If you want empty array for right hemi np.array([], int)
vertices = [stc.lh_vertno, stc.rh_vertno]
stc_feat = mne.SourceEstimate(np.abs(patterns), vertices=vertices,
                              tmin=stc.tmin, tstep=stc.tstep, subject=cfg.current_subject)

brain = stc_feat.plot(views=['lat'], transparent=True,
                      initial_time=0.1, time_unit='s',
                      subjects_dir=cfg.patients_path)
