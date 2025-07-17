import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import pickle
import pandas as pd
from tqdm import tqdm

from region_select import *
from config import PATH_DICT

sys.path.append(PATH_DICT['repo'])

# Region specific plotting arguments
ylabels = {
    'M1_psid': 'Velocity Prediction ' + r'$r^2$',
    'S1_psid': 'Velocity Prediction ' + r'$r^2$',
    'HPC_peanut': 'Position Prediction' + r'$r^2$',
    'VISp': 'Classification Accuracy'
}

# Taking a 1/3 of the average ambient unit count
auc_dim_dict = {
    'M1_psid': 48,
    'S1_psid': 52,
    'HPC_peanut': 20,
    'VISp': 26
}

diff_yticks = {
    'M1_psid': [0., 0.12],
    'S1_psid': [0., 0.1],
    'HPC_peanut': [0, 0.25],
    'VISp': [0, 0.05],
}

diff_ylims = {
    'M1_psid': [0, 0.13],
    'S1_psid': [0., 0.11],
    'HPC_peanut': [0, 0.25],
    'VISp': [0, 0.052]
}

xlim_dict = {
    'M1_psid': [1,48],
    'S1_psid': [1,52],
    'HPC_peanut': [1,20],
    'VISp': [1,26]
}

xtick_dict = {
    'M1_psid': [1, 15, 30, 45],
    'S1_psid': [1, 25, 50],
    'HPC_peanut': [1, 10, 20],
    'VISp': [1, 13, 26]
}

ytick_dict = {
    'M1_psid': [0, 0.25, 0.5],
    'S1_psid': [0, 0.125, 0.25],
    'HPC_peanut': [0, 0.3, 0.6],
    'VISp': [0, 0.2, 0.4]
}

inset_locs = {
    'M1_psid':[0.85, 0.1, 0.35, 0.35],
    'S1_psid':[0.95, 0.05, 0.35, 0.35],
    'HPC_peanut':[0.95, 0.05, 0.35, 0.35],
    'VISp': [0.85, 0.08, 0.35, 0.35]
}

from region_select import loader_kwargs

def get_decoding_performance(df, region, **kwargs):
    if region in ['M1_psid', 'S1_psid']:
        return df.iloc[0]['r2'][1]
    elif region in ['M1_psid_rand', 'S1_psid_rand']:
        return np.array([df.iloc[k]['r2'][1] for k in range(df.shape[0])])
    elif region in ['M1_psid_sup', 'S1_psid_sup']:
        return df.iloc[0]['r2'][1][kwargs['dim_index']]
    elif region == 'HPC_peanut':
        if np.isscalar(df.iloc[0]['r2']):   
            return df.iloc[0]['r2']
        else:
            return df.iloc[0]['r2'][0]
    elif region == 'HPC_peanut_rand':
        return np.array([df.iloc[k]['r2'][0] for k in range(df.shape[0])])
    elif region == 'HPC_peanut_sup':
        # Return the asymptotic (i.e. rank 2) position performance if dim_index is greater than 0
        if kwargs['dim_index'] > 0:
            return df.iloc[0]['r2'][0][-1]
        else:
            return df.iloc[0]['r2'][0][0]
    elif region in ['VISp']:
        return 1 - df.iloc[0]['loss']
    elif region in ['VISp_rand']:
        return np.array([1 - df.iloc[k]['r2'][0]['loss'] for k in range(df.shape[0])])
    elif region in ['VISp_sup']:
        dims = [df.iloc[k]['dim'] for k in range(df.shape[0])]
        acc = [df.iloc[k]['acc'] for k in range(df.shape[0])]
        # Make sure dims are in ascending order already in the dataframe
        assert(np.allclose(dims, np.sort(dims)))
        return acc[kwargs['dim_index']]
    else:
        raise NotImplementedError

def calc_ylim(ax):
    lines = ax.get_lines()
    
    # Extract y data from all plotted elements
    y_data = []
    for line in lines:
        y_data.extend(line.get_ydata())
        
    # Calculate the max y value and set ylim to 1.05 times that
    if y_data:
        max_y = np.nanmax(y_data)
    else:
        raise ValueError('No y data found in the plot')
    return [0, 1.05 * max_y]

def calc_yticks(ylim):
    # Calculate yticks with 3 values: 0, middle, and near max
    max_y = ylim[1]
    
    # Round max_y to the nearest multiple of 0.05
    max_tick = round(max_y / 0.1) * 0.1
    
    # Calculate middle tick as half of max_tick
    middle_tick = round(max_tick / 2, 2)
    
    # Ensure we have exactly 3 ticks
    yticks = [0, middle_tick, max_tick]
    
    return yticks


def inset_calculation(df, region, session_key):

    auc_dim = auc_dim_dict[region]
    if region in ['M1_psid', 'S1_psid']:
        # Disregard the df and instead load high d decoding df
        if region == 'M1_psid':
            with open(PATH_DICT['df'] + '/sabes_highd_decoding_df.pkl', 'rb') as f:
                df = pickle.load(f)
            df = pd.DataFrame(df)   
        else:
            with open(PATH_DICT['df'] + '/sabes_highd_decoding_dfS1.pkl', 'rb') as f:
                df = pickle.load(f)
            df = pd.DataFrame(df)
    sessions = np.unique(df[session_key].values)
    dims = np.arange(1, auc_dim + 1)
    # Get fca/pca r2s
    r2fc = np.zeros((len(sessions), dims.size, folds.size))
    r2pca = np.zeros((len(sessions), dims.size, folds.size))

    for i, session in tqdm(enumerate(sessions)):
        for j, dim in enumerate(dims):               
            for f in folds:
                df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                dim_fold_df = apply_df_filters(df, **df_filter)
                assert(dim_fold_df.shape[0] == 1)
                r2fc[i, j, f] = get_decoding_performance(dim_fold_df, region)
                df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 
                            'dimreduc_method':'PCA'}
                pca_df = apply_df_filters(df, **df_filter)
                assert(pca_df.shape[0] == 1)
                r2pca[i, j, f] = get_decoding_performance(pca_df, region)

    # Sum
    r2fc = np.mean(r2fc, axis=2)
    r2pca = np.mean(r2pca, axis=2)

    pca_auc = np.sum(r2pca, axis=1)
    fca_auc = np.sum(r2fc, axis=1)
    return pca_auc, fca_auc

if __name__ == '__main__':

    regions = ['M1_psid', 'S1_psid', 'HPC_peanut', 'VISp']

    include_rand_control = True
    nrand = 1000
    include_supervised_ub = True
    include_inset = True

    # Where to save?
    if len(sys.argv) > 1:   
        figpath = sys.argv[1]
    else:
        figpath = PATH_DICT['figs']

    r2p_across_regions = []
    r2f_across_regions = []
    r2_sup_across_regions = []
    sessions_per_region = []
    for region in regions:

        loader_kwargs[region]['use_highd'] = True

        df, session_key = load_decoding_df(region, **loader_kwargs[region])        
        sessions = np.unique(df[session_key].values)
        sessions_per_region.append(sessions)
        dims = np.unique(df['dim'].values)
        folds = np.unique(df['fold_idx'].values)
        r2fc = np.zeros((len(sessions), dims.size, folds.size))
        r2pca = np.zeros((len(sessions), dims.size, folds.size))
    
        # assert(max(dims) == 30)
   
        if include_rand_control:
            if not os.path.exists(PATH_DICT['tmp'] + '/rand_decoding_%s_rand.pkl' % region):
                df_rand = load_rand_decoding_df(region, **loader_kwargs[region])                
                compute_rand = True
                dims_rand = np.unique(df_rand['dim'].values)
                r2_rand = np.zeros((len(sessions), dims_rand.size, folds.size, nrand))
            else:
                compute_rand = False
                with open(PATH_DICT['tmp'] + '/rand_decoding_%s_rand.pkl' % region, 'rb') as f:
                    r2_rand = pickle.load(f)
                    try:
                        dims_rand = pickle.load(f)
                    except:
                        dims_rand = dims
        else:
            compute_rand = False
        
        if include_supervised_ub:
            df_sup, dims_sup = load_supervised_decoding_df(region, **loader_kwargs[region])
            if dims_sup is None:
                dims_sup = dims
            r2_sup = np.zeros((len(sessions), dims_sup.size, folds.size))

        for i, session in tqdm(enumerate(sessions)):
            for j, dim in enumerate(dims):               
                for f in folds:
                    df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                    dim_fold_df = apply_df_filters(df, **df_filter)
                    assert(dim_fold_df.shape[0] == 1)
                    r2fc[i, j, f] = get_decoding_performance(dim_fold_df, region)
                    df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 
                                'dimreduc_method':'PCA'}
                    pca_df = apply_df_filters(df, **df_filter)
                    assert(pca_df.shape[0] == 1)
                    r2pca[i, j, f] = get_decoding_performance(pca_df, region)

            if include_supervised_ub:
                for f in folds:
                    df_filter = {session_key:session, 'fold_idx':f}
                    sup_df = apply_df_filters(df_sup, **df_filter)
                    for j, dim in enumerate(dims_sup):
                        r2_sup[i, j, f] = get_decoding_performance(sup_df, 
                                                                    region + '_sup', dim_index=j)

            if include_rand_control and compute_rand:
                for j, dim in enumerate(dims_rand):
                    for f in folds:
                        df_filter = {'dim':dim, session_key:session, 'fold_idx':f}
                        rand_df = apply_df_filters(df_rand, **df_filter)
                        # Perhaps this dim value was not included
                        if rand_df.shape[0] == 0:
                            r2_rand[i, j, f] = np.nan
                        else:
                            rand_performance = get_decoding_performance(rand_df, region + '_rand')
                            if rand_performance.size > nrand:
                                r2_rand[i, j, f] = rand_performance[0:nrand]
                                print(f'Region {region} has {rand_performance.size} random decoding values for dim {dim} and session {session} and fold {f}')
                            else:
                                r2_rand[i, j, f] = rand_performance


        # Rand assemblage takes time so save it as tmp
        if compute_rand:
            with open(PATH_DICT['tmp'] + '/rand_decoding_%s_rand.pkl' % region, 'wb') as f:
                f.write(pickle.dumps(r2_rand))
                f.write(pickle.dumps(dims_rand))

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        colors = ['black', 'red', '#781820', '#8a4be3']
        dim_vals = dims
        n = len(sessions)
        # FCCA averaged over folds
        fca_r2 = np.mean(r2fc, axis=2)
        # PCA
        pca_r2 = np.mean(r2pca, axis=2)

        ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(n),
                        np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(n), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

        ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(n),
                        np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(n), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])

        if include_rand_control:
            # Average over folds and nrand, and then show the std err across sessions
            r2_rand_ = np.nanmean(r2_rand, axis=(-1, -2))
            yavg = np.mean(r2_rand_, axis=0)
            ystd = np.std(r2_rand_, axis=0)
            ax.fill_between(dims_rand, yavg + ystd/np.sqrt(n),
                            yavg - ystd/np.sqrt(n), 
                            color=colors[2], alpha=0.25)
            ax.plot(dims_rand, yavg, color=colors[2])

        if include_supervised_ub:
            # Average over folds
            r2_sup = np.mean(r2_sup, axis=2)
            r2_means = np.mean(r2_sup, axis=0)
            r2_sems = np.std(r2_sup, axis=0) / np.sqrt(n)
            
            # Plot mean ± SE
            ax.fill_between(dims_sup, 
                            r2_means + r2_sems,
                            r2_means - r2_sems,
                            color=colors[3], alpha=0.25)
            ax.plot(dims_sup, r2_means, color=colors[3])
            r2_sup_across_regions.append((r2_sup, r2_sems))

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(ylabels[region], fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        xlim = xlim_dict[region]
        ax.set_xlim(xlim)
        ax.set_xticks(xtick_dict[region])

        # Set ylims to 1.05 times the max of the data that is plotted
        # Assume that the supervised decoding provides this bound
        if include_supervised_ub:
            r2_sup_plus_sem = r2_means + r2_sems
            # Truncate to the dimensions being plotted
            max_dim = max(xlim)
            ymax = np.max(r2_sup_plus_sem[dims_sup <= max_dim])
            ylim = [0, 1.05 * ymax]
            ax.set_ylim(ylim)
            ax.set_yticks(ytick_dict[region])
        # Add legend manually
        # ax.legend(['FBC', 'FFC'], fontsize=10, loc='upper left', frameon=False)

        if include_inset:
            axin = ax.inset_axes(inset_locs[region])

            # Get the AUCs. 
            pca_auc, fca_auc = inset_calculation(df, region, session_key)

            # Run a signed rank test
            _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')
            print('Across session WCSRT: %f' % p)
            print('Sum Delta AUC mean: %f' % np.mean(fca_auc - pca_auc))
            se = np.std(fca_auc - pca_auc)/np.sqrt(float(n))
            print(f'Sum Delta AUC SE: {se}')
            axin.scatter(np.zeros(n), pca_auc, color='k', alpha=0.75, s=3)
            axin.scatter(np.ones(n), fca_auc, color='r', alpha=0.75, s=3)
            axin.plot(np.array([(0, 1) for _ in range(pca_r2.shape[0])]).T, 
                      np.array([(y1, y2) for y1, y2 in zip(pca_auc, 
                                                           fca_auc)]).T, color='k', alpha=0.5)
            axin.set_yticks([])
            # axin.set_ylabel('Decoding AUC', fontsize=10)
            axin.set_xlim([-0.5, 1.5])
            axin.set_xticks([0, 1])
            # axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)

            fname = '%s/%s_decodingvdim.pdf' % (figpath, region)

        fig.savefig(fname, bbox_inches='tight', pad_inches=0)


        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(n),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(n), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')

        # For each recording session within each region, find the dimension of peak r^2 difference
        dr2 = fca_r2 - pca_r2
        dr2 = dr2[:, 3:]
        # Find the dimension of peak r^2 difference
        max_dr2 = np.max(dr2, axis=1)
        max_dr2_idx = np.argmax(dr2, axis=1)
        max_dr2_dim = dim_vals[max_dr2_idx + 3]
        pca_performance = np.array([pca_r2[i, max_dr2_idx[i] + 3] for i in range(n)])
        fractional_dr2 = np.divide(max_dr2, pca_performance)

        print('Mean fractional improvement: %f' % np.mean(fractional_dr2))
        print('SE fractional improvement: %f' % float(np.std(fractional_dr2)/float(np.sqrt(n))))
        print('Mean dimension of peak improvement: %f' % np.mean(max_dr2_dim))
        print('All peak dimensions: %s' % str(max_dr2_dim))
        print('All fractional improvements: %s' % str(fractional_dr2))

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' ' + ylabels[region], fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_xlim(xlim)
        ax.set_xticks(xtick_dict[region])

        # Similarly, set ylim to 1.05 times the max of the data that is plotted
        max_trace = np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(n)
        max_dim = max(xlim)
        ymax = np.max(max_trace[dims <= max_dim])
        ylim = [0, 1.05 * ymax]
        ax.set_ylim(ylim)
        ax.set_yticks(diff_yticks[region])

        fname = '%s/%s_decoding_delta.pdf' % (figpath, region)

        fig.savefig(fname, bbox_inches='tight', pad_inches=0)

        r2f_across_regions.append(fca_r2)
        r2p_across_regions.append(pca_r2)

    with open(PATH_DICT['tmp'] + '/decodingvdim_across_regions.pkl', 'wb') as f:
        pickle.dump(r2f_across_regions, f)
        pickle.dump(r2p_across_regions, f)
        pickle.dump(r2_sup_across_regions, f)
        pickle.dump(regions, f)


        # Summary statistics    
        # dr2 = np.divide(fca_r2 - pca_r2, pca_r2)
        # print('Mean Peak Fractional improvement: %f' % np.mean(np.max(dr2, axis=-1)))
        # # print('S.E. Fractional improvement: %f' % )
        # se = np.std(np.max(dr2, axis=-1))/np.sqrt(dr2.shape[0])
        # print('S.E. Peak Fractional improvement: %f' % se)