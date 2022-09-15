import numpy as np
import matplotlib.pyplot as plt
import mne
from seeg_action import project_config as cfg


if __name__ == '__main__':
    # Take arguments and initialize configurations
    parser = cfg.init_argparse()
    args = parser.parse_args()
    cfg.init_config(args.subject_name)

    mne.set_log_level('ERROR')

    if cfg.epochs_file.exists():
        epochs = mne.read_epochs(cfg.epochs_file)


    epochs.pick_types(seeg=True)

    montage = epochs.get_montage()
    head_mri_t = mne.read_trans(cfg.subject_head_mri_t)
    montage.apply_trans(mne.transforms.invert_transform(head_mri_t))

    labels, colors = mne.get_montage_volume_labels(
        montage, cfg.current_subject, subjects_dir=cfg.patients_path, aseg='aseg')

    wm_channels = [c for c, l in labels.items() if len(l) > 0 and "White-Matter" in l[0]]
    epochs.drop_channels(wm_channels)


    epochs_action = epochs['ST']
    for cond in epochs_action.event_id:
        action_class, _, action_examplar, object_size, actor = cond.split('/')
        evk_cs = epochs['/'.join([action_class, 'CS', object_size, actor])].average()
        evk_cd = epochs['/'.join([action_class, 'CD', object_size, actor])].average()
        evk_control = mne.combine_evoked([evk_cs, evk_cd], [0.5, 0.5])
        epochs_action[cond].subtract_evoked(evk_control)

    del epochs
    #for channel in epochs_action.ch_names:
    for channel in ["V'1", "V'9", "Z12", "E3", "Z'11", "C'2"]:
        epochs_ch = epochs_action.copy().pick(channel)

        power = []
        freqs = np.geomspace(5, 152, num=50)
        n_cycles = freqs / freqs[0]
        tfr_tmin, tfr_tmax = -0.3, 2.3
        for action_class in cfg.action_classes:
            e = epochs_ch[action_class]
            epochs_tfr = mne.time_frequency.tfr_morlet(e, freqs, n_cycles=n_cycles,
                                                       average=False, return_itc=False, n_jobs=-3)
            epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0))
            epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)
            power.append(epochs_tfr.data.squeeze())

        F_obs, clusters, cluster_p_values, H0 = \
            mne.stats.permutation_cluster_test(power, n_jobs=-3)

        # Check if there is any significant cluster
        if np.any(cluster_p_values < .05):
            print('Found cluster in ', channel)

            good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
            fig, ax = plt.subplots()
            F_obs_plot = F_obs.copy()
            F_obs_plot_sig = np.zeros(F_obs_plot.shape) * np.nan
            for i_clu, clu_idx in enumerate(good_cluster_inds):
                # unpack cluster information, get unique indices
                freq_inds, time_inds = clusters[clu_idx]
                time_inds = np.unique(time_inds)
                freq_inds = np.unique(freq_inds)
                F_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = \
                    F_obs_plot[tuple(np.meshgrid(freq_inds, time_inds))]

            for f_image, cmap in zip([F_obs_plot, F_obs_plot_sig], ['gray', 'autumn']):
                c = ax.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                              extent=[tfr_tmin, tfr_tmax,
                                      freqs[0], freqs[-1]])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('Clusters in ' + channel)
            plt.show()
        else:
            print('No cluster in ', channel)




