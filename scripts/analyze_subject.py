import mne
from seeg_action import project_config as cfg
from seeg_action import preprocessing as pp
from seeg_action import mass_univariate_analysis as mua


if __name__ == '__main__':
    # Take arguments and initialize configurations
    parser = cfg.init_argparse()
    args = parser.parse_args()
    cfg.init_config(args.subject_name)

    if cfg.epochs_file.exists():
        epochs = mne.read_epochs(cfg.epochs_file)
        epochs.pick_types(seeg=True)

        epochs.drop_channels(pp.get_wm_channels(epochs.info))
        epochs.pick(["V'4"])
        mua.mass_planned_contrast_anova(epochs)
