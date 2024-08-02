import numpy as np
import matplotlib.pyplot as plt
import mne
from seeg_action import project_config
from seeg_action import preprocessing as pp
import sys


def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


if __name__ == '__main__':
    # Take arguments and initialize configurations

    parser = project_config.init_argparse()
    args = parser.parse_args()
    cfg = project_config.ProjectConfig(args.subject_name)

    mne.set_log_level('ERROR')
    draw = False

    epochs = mne.read_epochs(cfg.epochs_file)
    epochs.pick_types(seeg=True)
    epochs.drop_channels(pp.get_wm_channels(epochs.info))

    freqs = np.geomspace(5, 152, num=50)
    n_cycles = freqs / freqs[0]
    tfr_tmin, tfr_tmax = -0.3, 2.3

    epochs_tfr = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles=n_cycles,
                                               average=False, return_itc=False, n_jobs=-3)
    epochs_tfr.apply_baseline(mode='zscore', baseline=(-0.4, 0))
    epochs_tfr.crop(tmin=tfr_tmin, tmax=tfr_tmax)

    def _cluster_statistics(data, comparison, show_plots=False):
        threshold_tfce = dict(start=0, step=0.2)
        F_obs, clusters, cluster_p_values, H0 = \
            mne.stats.permutation_cluster_test(data, threshold=threshold_tfce, n_jobs=-3)

        # Check if there is any significant cluster
        if np.any(cluster_p_values < .05):
            print(f'Found cluster in {channel} for {comparison}')
            if show_plots:
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
                ax.set_title(f'Clusters in {channel} for {comparison}')
                plt.show()
        #else:
        #    print('No cluster in ', channel, ' for ', title)

    print()
    for channel in progressbar(epochs.ch_names, 'Analyzing: ', 100):
        power = {}
        for action_class in cfg.action_classes:
            power[action_class] = epochs_tfr[action_class].pick(channel).data.squeeze()

        _cluster_statistics([power['BOD'], power['OBJ'], power['PER']], comparison='BOD vs OBJ vs PER')
        #_cluster_statistics([power['BOD'], power['OBJ']], comparison='BOD vs OBJ')
        #_cluster_statistics([power['PER'], power['OBJ']], comparison='PER vs OBJ')
        #_cluster_statistics([power['BOD'], power['PER']], comparison='BOD vs PER')