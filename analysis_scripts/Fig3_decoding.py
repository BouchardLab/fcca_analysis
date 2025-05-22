import pdb
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
import itertools

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from region_select import *
from config import PATH_DICT

sys.path.append(PATH_DICT['repo'])
from utils import calc_loadings

from loaders import load_peanut
from decoders import lr_decoder

# Region specific plotting arguments
ylabels = {
    'M1': 'Velocity Prediction ' + r'$r^2$',
    'M1_psid': 'Velocity Prediction ' + r'$r^2$',
    'M1_trialized': 'Velocity Prediction ' + r'$r^2$',
    'S1': 'Velocity Prediction ' + r'$r^2$',
    'S1_psid': 'Velocity Prediction ' + r'$r^2$',
    'HPC_peanut': 'Position Prediction' + r'$r^2$',
    'HPC': 'Position Prediction' + r'$r^2$',
    'M1_maze': 'Velocity Prediction' + r'$r^2$',
    'ML': 'Classification Accuracy',
    'AM': 'Classification Accuracy',
    'mPFC': 'Position Prediction' + r'$r^2$',
    'VISp': 'Classification Accuracy'
}

# Taking a 1/3 of the average ambient unit count
auc_dim_dict = {
    'M1_psid': 48,
    'S1_psid': 52,
    'HPC_peanut': 20,
    'VISp': 26
}

# auc_dim_dict = {
#     'M1_psid': 30,
#     'S1_psid': 30,
#     'HPC_peanut': 30,
#     'VISp': 30,
#     'M1_maze': 30
# }


diff_yticks = {
    'M1': [0., 0.12],
    'M1_psid': [0., 0.12],
    'M1_trialized': [0., 0.06],
    'S1': [0., 0.1],
    'S1_psid': [0., 0.1],
    'HPC_peanut': [0, 0.25],
    'HPC': [0, 0.20],
    'M1_maze':[0, 0.06],
    'ML': [0, 0.12],
    'AM': [0, 0.12],
    'mPFC': [0, 0.1],
    'VISp': [0, 0.05],
}

diff_ylims = {
    'M1': [0, 0.125],
    'M1_psid': [0, 0.13],
    'M1_trialized': [0, 0.06],
    'S1': [0., 0.11],
    'S1_psid': [0., 0.11],
    'HPC_peanut': [0, 0.25],
    'HPC': [0, 0.20],
    'M1_maze': [0, 0.06],
    'ML': [-0.042, 0.125],
    'AM': [-0.01, 0.125],
    'mPFC': [-0.1, 0.1],
    'VISp': [0, 0.052]
}

def get_xlim_dict(match_dims_to_auc_dim):
    xlim_dict = {
        'M1':[1, 30],
        'M1_psid':[1, 30],
        'M1_trialized':[1, 30],
        'S1': [1, 30],
        'S1_psid': [1, 30],
        'HPC_peanut': [1, 30],
        'HPC': [1, 30],
        'M1_maze':[1, 30],
        'ML':[1, 59],
        'AM':[1, 59],
        'mPFC': [1, 30],
        'VISp':[1,30]
    }

    if match_dims_to_auc_dim:
        xlim_dict['M1_psid'] = [1,48]
        xlim_dict['S1_psid'] = [1,52]
        xlim_dict['HPC_peanut'] = [1,20]
        xlim_dict['VISp'] = [1,26]
    return xlim_dict

def get_xtick_dict(match_dims_to_auc_dim):
    xtick_dict = {
        'M1':[1, 15, 30],
        'M1_psid':[1, 15, 30],
        'M1_trialized':[1, 15, 30],
        'S1':[1, 15, 30],
        'S1_psid':[1, 15, 30],
        'HPC_peanut':[1, 15, 30],
        'HPC':[1, 15, 30],
        'M1_maze':[1, 15, 30],
        'ML':[1, 25, 50],
        'AM':[1, 25, 50],
        'mPFC': [1, 15, 30],
        'VISp': [1, 15, 30]
    }

    if match_dims_to_auc_dim:
        xtick_dict['M1_psid'] = [1, 15, 30, 45]
        xtick_dict['S1_psid'] = [1, 25, 50]
        xtick_dict['HPC_peanut'] = [1, 10, 20]
        xtick_dict['VISp'] = [1, 13, 26]
    return xtick_dict

ytick_dict = {
    'M1_psid': [0, 0.25, 0.5],
    'S1_psid': [0, 0.125, 0.25],
    'HPC_peanut': [0, 0.3, 0.6],
    'VISp': [0, 0.2, 0.4]
}

inset_locs = {
    'M1':[0.6, 0.1, 0.35, 0.35],
    'M1_psid':[0.85, 0.1, 0.35, 0.35],
    'M1_trialized':[0.6, 0.1, 0.35, 0.35],
    'S1':[0.8, 0.1, 0.35, 0.35],
    'S1_psid':[0.95, 0.05, 0.35, 0.35],
    'HPC_peanut':[0.95, 0.05, 0.35, 0.35],
    'HPC':[0.6, 0.1, 0.35, 0.35],
    'M1_maze': [0.1, 0.65, 0.35, 0.35],
    'ML': [0.6, 0.1, 0.35, 0.35],
    'AM': [0.68, 0.08, 0.35, 0.35],
    'mPFC': [0.68, 0.08, 0.35, 0.35],
    'VISp': [0.85, 0.08, 0.35, 0.35]

}

from region_select import loader_kwargs

def get_decoding_performance(df, region, **kwargs):
    if region in ['M1', 'S1', 'M1_trialized', 'M1_psid', 'S1_psid']:
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
    elif region == 'M1_maze':
        return df.iloc[0]['r2'][1]
    elif region in ['ML', 'AM']:
        return 1 - df.iloc[0]['loss']
    elif region in ['ML_sup', 'AM_sup']:
        return 1 - df.iloc[0]['r2'][0]['loss']
    elif region in ['ML_rand', 'AM_rand']:
        pdb.set_trace()
    elif region in ['mPFC', 'HPC']:
        return df.iloc[0]['r2'][0]
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
    # collections = ax.collections
    
    # Extract y data from all plotted elements
    y_data = []
    for line in lines:
        y_data.extend(line.get_ydata())
    
    # for collection in collections:
    #     if hasattr(collection, 'get_paths') and len(collection.get_paths()) > 0:
    #         for path in collection.get_paths():
    #             y_data.extend(path.vertices[:, 1])
    
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
                try:
                    assert(dim_fold_df.shape[0] == 1)
                except:
                    pdb.set_trace()
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


def statement_plot():
    # Called main script once and saved these
    # with open(PATH_DICT['tmp'] + '/decodingvdim_across_regions.pkl', 'wb') as f:
    #     pickle.dump(r2f_across_regions, f)
    #     pickle.dump(r2p_across_regions, f)
    #     pickle.dump(regions, f)

    with open(PATH_DICT['tmp'] + '/decodingvdim_across_regions.pkl', 'rb') as f:
        r2f_across_regions = pickle.load(f)
        r2p_across_regions = pickle.load(f)
        r2_sup_across_regions = pickle.load(f)
        regions = pickle.load(f)

    # Plot for AK research proposal - plot the percent of supervised r2 achieved for
    # M1/S1

    m1_idx = regions.index('M1_psid')
    s1_idx = regions.index('S1_psid')

    r2f_across_regions = [r2f_across_regions[m1_idx], r2f_across_regions[s1_idx]]
    r2p_across_regions = [r2p_across_regions[m1_idx], r2p_across_regions[s1_idx]]
    r2_sup_across_regions = [r2_sup_across_regions[m1_idx], r2_sup_across_regions[s1_idx]]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax2 = ax.twinx()

    dim_cutoff = 2

    for i in range(2):
        pct_supervised_fca = np.divide(r2f_across_regions[i], r2_sup_across_regions[i])
        pct_supervised_pca = np.divide(r2p_across_regions[i], r2_sup_across_regions[i])
        dims = np.arange(1, pct_supervised_fca.shape[1] + 1)[dim_cutoff:]

        mu1 = np.mean(pct_supervised_fca, axis=0)
        mu2 = np.mean(pct_supervised_pca, axis=0)
        se1 = np.std(pct_supervised_fca, axis=0) / np.sqrt(len(pct_supervised_fca))
        se2 = np.std(pct_supervised_pca, axis=0) / np.sqrt(len(pct_supervised_pca))

        mu1 = mu1[dim_cutoff:]
        mu2 = mu2[dim_cutoff:]
        se1 = se1[dim_cutoff:]
        se2 = se2[dim_cutoff:]

        ax.plot(dims, mu1, color='r')
        ax.fill_between(dims, mu1 + se1, mu1 - se1, color='r', alpha=0.25)
        ax2.plot(dims, mu2, color='k')
        ax2.fill_between(dims, mu2 + se2, mu2 - se2, color='k', alpha=0.25)

    ax.set_xlabel('Dimension', fontsize=14)
    # ax.set_ylabel('Improvement')

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    if save_plots:
        fig.savefig('./research_statement_plot.pdf', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':

    # regions = ['M1_psid', 'S1_psid', 'HPC_peanut', 'VISp']
    # regions = ['M1_psid', 'S1_psid', 'HPC_peanut', 'M1_maze']
    regions = ['M1_psid', 'S1_psid']

    include_rand_control = True
    nrand = 1000
    include_supervised_ub = True
    include_inset = True
    # Should we plot the decoding v. dim curves over the same dimension range as the we 
    # do the AUC statistical test on?
    match_dims_to_auc_dim = True
    save_plots = True

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

        # Flag to use the dataframes that include data for higher dimensions
        if match_dims_to_auc_dim:
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
                    try:
                        assert(dim_fold_df.shape[0] == 1)
                    except:
                        pdb.set_trace()
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

        # print(np.mean(fca_r2, axis=0))
        # print(np.mean(pca_r2, axis=0))

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
        xlim = get_xlim_dict(match_dims_to_auc_dim)[region]
        ax.set_xlim(xlim)
        ax.set_xticks(get_xtick_dict(match_dims_to_auc_dim)[region])

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

            # Get the AUCs. Calculate it in a more principled way, summing across
            # half the average number of units
            pca_auc, fca_auc = inset_calculation(df, region, session_key)

            # pca_auc = np.sum(pca_r2, axis=1)
            # fca_auc = np.sum(fca_r2, axis=1)
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

        if match_dims_to_auc_dim:
            fname = '%s/%s_decodingvdim_match_dims.pdf' % (figpath, region)
        else:
            fname = '%s/%s_decodingvdim.pdf' % (figpath, region)

        if save_plots:
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)


        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(n),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(n), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')

        # For each recording session within each region, find the dimension of peak r^2 difference
        # Enforce d > 3.
        dr2 = fca_r2 - pca_r2
        # Trim off dimensions < 3
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

        #ax.vlines(6, 0, np.mean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
        #ax.hlines(np.mean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
        xlim = get_xlim_dict(match_dims_to_auc_dim)[region]
        ax.set_xlim(xlim)
        ax.set_xticks(get_xtick_dict(match_dims_to_auc_dim)[region])

        # Similarly, set ylim to 1.05 times the max of the data that is plotted
        max_trace = np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(n)
        max_dim = max(xlim)
        ymax = np.max(max_trace[dims <= max_dim])
        ylim = [0, 1.05 * ymax]
        ax.set_ylim(ylim)
        ax.set_yticks(diff_yticks[region])

        if match_dims_to_auc_dim:
            fname = '%s/%s_decoding_delta_match_dims.pdf' % (figpath, region)
        else:
            fname = '%s/%s_decoding_delta.pdf' % (figpath, region)

        if save_plots:
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)

        r2f_across_regions.append(fca_r2)
        r2p_across_regions.append(pca_r2)

        # Summary statistics    
        # dr2 = np.divide(fca_r2 - pca_r2, pca_r2)
        # print('Mean Peak Fractional improvement: %f' % np.mean(np.max(dr2, axis=-1)))
        # # print('S.E. Fractional improvement: %f' % )
        # se = np.std(np.max(dr2, axis=-1))/np.sqrt(dr2.shape[0])
        # print('S.E. Peak Fractional improvement: %f' % se)


    with open(PATH_DICT['tmp'] + '/decodingvdim_across_regions.pkl', 'wb') as f:
        pickle.dump(r2f_across_regions, f)
        pickle.dump(r2p_across_regions, f)
        pickle.dump(r2_sup_across_regions, f)
        pickle.dump(regions, f)



    if len(regions) == 2:
        # Summary statistics region comparison            
        _, p1 = scipy.stats.mannwhitneyu(np.sum(r2f_across_regions[0], axis=1, keepdims=True),
                                         np.sum(r2f_across_regions[1], axis=1, keepdims=True),
                                         alternative='greater', axis=0)

        _, p2 = scipy.stats.mannwhitneyu(np.sum(r2p_across_regions[0], axis=1, keepdims=True),
                                         np.sum(r2p_across_regions[1], axis=1, keepdims=True),
                                         alternative='greater', axis=0)

    else:

        try:
            # Comparison between M1 and its trialized version
            m1_trialized_idx = regions.index('M1_trialized')
            m1_idx = regions.index('M1_psid')
            m1_maze_idx = regions.index('M1_maze')
        except ValueError:
            sys.exit()

        


        # M1 trialized is missing a session...
        common_session_indices = [i for i, elem in enumerate(sessions_per_region[m1_idx])
                                  if elem in sessions_per_region[m1_trialized_idx]]
        
        common_session_indices = np.array(common_session_indices)

        # First comparison - difference between M1/M1 trialized at d=30
        fr1 = r2f_across_regions[m1_idx][common_session_indices]
        fr2 = r2f_across_regions[m1_trialized_idx]
        fr3 = r2f_across_regions[m1_maze_idx]

        pr1 = r2p_across_regions[m1_idx][common_session_indices]
        pr2 = r2p_across_regions[m1_trialized_idx]
        pr3 = r2f_across_regions[m1_maze_idx]

        _, p1 = scipy.stats.wilcoxon(fr1[:, -1], fr2[:, -1], alternative='greater')
        _, p2 = scipy.stats.wilcoxon(pr1[:, -1], pr2[:, -1], alternative='greater')

        print(f'FBC M1 vs. M1 trialized p = {p1}, FFC p = {p2}')

        # Second comparison - difference between difference in FBC/FFC at d=6
        dr1 = fr1 - pr1
        dr2 = fr2 - pr2

        _, p3 = scipy.stats.wilcoxon(dr1, dr2, alternative='greater')
        print(f'Delta decoding p = {p3}')
