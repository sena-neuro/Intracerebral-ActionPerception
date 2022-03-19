import mne
from src import project_config as cfg
from src import io
from src import preprocessing as pp
from src import artifact_rejection as ar


if __name__ == '__main__':
    mne.viz.set_browser_backend('pyqtgraph')

    # Take arguments and initialize configurations
    parser = cfg.init_argparse()
    args = parser.parse_args()

    cfg.set_var('current_subject', args.subject_name)
    cfg.set_var('redo', args.redo)

    # Read raw annotated EEG
    raw_fif_file = cfg.steps_save_path / f'{args.subject_name}_STEP_0_annotated_raw.fif'
    if raw_fif_file.exists() and not args.redo:
        raw = mne.io.read_raw_fif(raw_fif_file)
    else:
        raw = io.nk_to_mne(args.subject_name)
    orig_raw = raw.copy()


    # %%
    # HFO
    raw = ar.reject_hfo(raw)

    # Epoching
    epochs_fif_file = cfg.steps_save_path / f'{args.subject_name}_STEP_1_epo.fif'
    if epochs_fif_file.exists() and not args.redo:
        epochs = mne.read_epochs(epochs_fif_file)
    else:
        epochs = pp.epoch(raw)

    mn_st_epochs = epochs['MN-ST']
    # mn_st_epochs.plot(scalings='auto', n_channels=10, n_epochs=5)

    # PLOT Mean potential over channels per trial
    # y-axis : epochs/trials
    # x-axis : time
    # color : mean mV over channels
    # mn_st_epochs.plot_image(picks='seeg', combine='mean')

    # PLOT Average Evoked Potential per channel
    # y-axis : epochs/trials
    # x-axis : time
    # color : average evoked potential
    # mn_st_evoked = mn_st_epochs.average()
    # mn_st_evoked.plot_image()