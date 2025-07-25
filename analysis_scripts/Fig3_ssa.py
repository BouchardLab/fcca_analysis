import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

from region_select import load_decoding_df, apply_df_filters, loader_kwargs
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])

dim = 6

if __name__ == '__main__':

    figpath = PATH_DICT['figs']
    regions = ['M1_psid', 'S1_psid', 'HPC_peanut', 'VISp']

    for region in tqdm(regions):
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        
        sessions = np.unique(df[session_key].values)
        ss_angles = np.zeros((len(sessions), 5, dim))
        folds = np.arange(5)
        dimvals = np.unique(df['dim'].values)
        for i, session in enumerate(sessions):
            for f, fold in enumerate(folds):
                df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                dfpca = apply_df_filters(df, **df_filter)
                df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                dffcca = apply_df_filters(df, **df_filter)

                assert(dfpca.shape[0] == 1)
                assert(dffcca.shape[0] == 1)
                
                ss_angles[i, f, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])

        fig, ax = plt.subplots(figsize=(1, 4))
        medianprops = {'linewidth':0}
        bplot = ax.boxplot(np.mean(ss_angles, axis=-1).ravel(), patch_artist=True, medianprops=medianprops, notch=True, vert=True, showfliers=False)
        ax.set_xticks([])
        ax.set_yticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
        ax.set_yticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'], fontsize=14)
        ax.yaxis.tick_right()
        ax.set_ylabel(r'FBC/FFC avg. $\theta$ (rads)', fontsize=14, rotation=90, labelpad=10)


        for patch in bplot['boxes']:
            patch.set_facecolor('k')
            patch.set_alpha(0.75)

        mean = np.mean(np.mean(ss_angles, axis=(1, 2)).ravel())
        print(f"Avg. subspace angle: {mean * 180/np.pi:.2f} deg")
        se = np.std(np.mean(ss_angles, axis=-1).ravel())/np.sqrt(ss_angles.shape[0])
        print(f'S.E. subspace angles: {se * 180/np.pi:.2f} deg')
        median = np.median(np.mean(ss_angles, axis=(1,2)).ravel())
        print(f'Median subspace angles: {median:.2f} rads, {median * 180/np.pi:.2f} deg')
        iqr25 = np.quantile(np.mean(ss_angles, axis=(1,2)).ravel(), 0.25)
        iqr75 = np.quantile(np.mean(ss_angles, axis=(1,2)).ravel(), 0.75)
        print(f'IQR subspace angles: ({iqr25:.2f}, {iqr75:.2f})')

        # ax.set_title('%s \n %s \n %s' % (la, deca, dra))
        fig.tight_layout()
        fig.savefig('%s/ss_angles%s.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)

        if False:
            # Also generate here supplementary figures that communicate the full spread of angles
            dimvals = np.unique(df['dim'].values)[:-1]
            ssa1_median = np.zeros((len(sessions), len(dimvals), 5))
            ssa1_min = np.zeros((len(sessions), len(dimvals), 5))
            ssa1_max = np.zeros((len(sessions), len(dimvals), 5))

            # d to d + 1 comparison
            ssa2_median = np.zeros((len(sessions), len(dimvals), 5))
            ssa2_min = np.zeros((len(sessions), len(dimvals), 5))
            ssa2_max = np.zeros((len(sessions), len(dimvals), 5))

            # Reference of what PCA looks like
            ssa3_mean = np.zeros((len(sessions), len(dimvals), 5))
            ssa3_min = np.zeros((len(sessions), len(dimvals), 5))
            ssa3_max = np.zeros((len(sessions), len(dimvals), 5))

            # SVD of concatenated projection
            joint_sv = np.zeros((len(sessions), len(dimvals), 5))


            for i, session in enumerate(sessions):
                for j, dim in enumerate(dimvals):
                    for f in range(5):
                        df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                        dfd1_fc = apply_df_filters(df, **df_filter)
                        df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                        dfd1_pc = apply_df_filters(df, **df_filter)

                        assert(dfd1_fc.shape[0] == 1)
                        assert(dfd1_pc.shape[0] == 1)

                        ssa1 = scipy.linalg.subspace_angles(dfd1_fc.iloc[0]['coef'], dfd1_pc.iloc[0]['coef'][:, 0:dim])
                        joint_proj = np.hstack([dfd1_fc.iloc[0]['coef'], dfd1_pc.iloc[0]['coef'][:, 0:dim]])
                        s = np.linalg.svd(joint_proj, compute_uv=False)
                        joint_sv[i, j, f] = np.sum(s)


                        if j+1 == len(dimvals): 
                            ssa2 = np.array([np.nan])
                            ssa3 = np.array([np.nan])
                        else:
                            df_filter = {'dim':dim + 1, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                            dfd2 = apply_df_filters(df, **df_filter)
                            assert(dfd2.shape[0] == 1)

                            ssa2 = scipy.linalg.subspace_angles(dfd1_fc.iloc[0]['coef'], dfd2.iloc[0]['coef'])

                            df_filter = {'dim':dim +1, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                            dfd2 = apply_df_filters(df, **df_filter)
                            assert(dfd2.shape[0] == 1)
                            ssa3 = scipy.linalg.subspace_angles(dfd1_pc.iloc[0]['coef'][:, 0:dim], dfd2.iloc[0]['coef'][:, 0:dim])

                        r = {}
                        r['session'] = session
                        r['dim'] = dim
                        r['fold'] = f
                        r['ssa1'] = ssa1
                        r['ssa2'] = ssa2

                        ssa1_median[i, j, f] = np.median(ssa1)
                        ssa1_min[i, j, f] = np.min(ssa1)
                        ssa1_max[i, j, f] = np.max(ssa1)

                        ssa2_median[i, j, f] = np.nanmedian(ssa2[0:dim])
                        ssa2_min[i, j, f] = np.nanmin(ssa2)
                        ssa2_max[i, j, f] = np.nanmax(ssa2)

                        ssa3_mean[i, j, f] = np.nanmean(ssa3[0:dim])
                        ssa3_min[i, j, f] = np.nanmin(ssa3)
                        ssa3_max[i, j, f] = np.nanmax(ssa3)

            # New plot of just the joint singular values
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(dimvals, np.mean(joint_sv, axis=(0, -1)), color='b')
            ax.plot(dimvals, 2*dimvals, color='k', linestyle='dashed')
            ax.set_ylabel('Effective Dimensionality of ' + r'$[V_{FBC}, V_{FFC}]$')
            ax.set_xlim([1, 30])
            ax.set_xticks([1, 15, 30])
            ax.set_xlabel('Dimension')
            fig.tight_layout()

            fig.savefig('%s/jointsv_vdim%s.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(np.mean(np.mean(ssa1_median, axis=0), axis=-1), color='b', alpha=0.75, linestyle='-')
            ax.plot(np.mean(np.mean(ssa1_min, axis=0), axis=-1), color='b', alpha=0.75, linestyle='--')
            ax.plot(np.mean(np.mean(ssa1_max, axis=0), axis=-1), color='b', alpha=0.75, linestyle=':')
            ax.set_ylim([0, np.pi/2])
            ax.set_yticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
            ax.set_yticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])

            ax.set_ylabel('FCCA/PCA subspace angle')
            ax.set_xlim([1, 30])
            ax.set_xticks([1, 15, 30])
            ax.set_xlabel('Dimension')
            ax.legend(['Median', 'Min', 'Max'])
            fig.tight_layout()
            fig.savefig('%s/ssa_vdim%s.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)


            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            ax.plot(np.mean(np.mean(ssa2_median, axis=0), axis=-1), color='b', alpha=0.75, linestyle='-')
            ax.plot(np.mean(np.mean(ssa2_min, axis=0), axis=-1), color='b', alpha=0.75, linestyle='--')
            ax.plot(np.mean(np.mean(ssa2_max, axis=0), axis=-1), color='b', alpha=0.75, linestyle=':')

                
            ax.set_ylabel('FCCA d/FCCA d + 1 subspace angle')
            ax.set_xlim([1, 30])
            ax.set_xticks([1, 15, 30])
            ax.set_xlabel('Dimension')
            ax.legend(['Median', 'Min', 'Max'])
            ax.set_yticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
            ax.set_yticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])

            # ax[1].plot(np.mean(np.mean(ssa3_mean, axis=0), axis=-1))
            # ax[1].plot(np.mean(np.mean(ssa3_min, axis=0), axis=-1))
            # ax[1].plot(np.mean(np.mean(ssa3_max, axis=0), axis=-1))
            fig.tight_layout()
            fig.savefig('%s/ssa_ddp1%s.pdf' % (figpath, region),
                        bbox_inches='tight', pad_inches=0)
