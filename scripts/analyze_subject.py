import mne
from seeg_action import project_config as cfg
from seeg_action import mass_univariate_analysis as mua


if __name__ == '__main__':
    # Take arguments and initialize configurations
    parser = cfg.init_argparse()
    args = parser.parse_args()
    cfg.init_config(args.subject_name)

    if cfg.epochs_file.exists():
        epochs = mne.read_epochs(cfg.epochs_file)
        results = mua.mass_planned_contrast_anova(epochs)
