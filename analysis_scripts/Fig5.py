import pdb
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy 
import pickle
import pandas as pd
# from statsmodels.stats import multitest


from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])
from utils import calc_loadings
from region_select import *
from Fig4 import get_loadings_df
from Fig3_decoding import get_decoding_performance

def get_su_calcs(region):
    if region in ['M1_psid', 'S1_psid']:
        region = region.split('_psid')[0]

    su_calcs_path = PATH_DICT['tmp'] + '/su_calcs_%s.pkl' % region
    with open(su_calcs_path, 'rb') as f:
        su_stats = pickle.load(f)

    su_calcs_df = pd.DataFrame(su_stats)
    return su_calcs_df

def get_marginal_dfs(region):
    # Fill in directories for marginals:
    root_path = PATH_DICT['df']

    if region in ['M1', 'S1']:
        region = region + '_psid'

    if region == 'M1_psid':
        with open(root_path + '/sabes_marginal_psid_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        df = df.iloc[filt]
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        marginals_df = pd.concat([df_pca, df_fcca])
    elif region == 'S1_psid':
        with open(root_path + '/sabes_marginal_psid_dfS1.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        marginals_df = df.iloc[filt]
    elif region == 'HPC_peanut':
        with open(root_path + '/peanut_marginal_decoding25_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        marginals_df = pd.DataFrame(rl)


        df_pca = apply_df_filters(marginals_df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(marginals_df, dimreduc_args={'T':5, 
              'loss_type':'trace', 'n_init':10, 'marginal_only':True})
        marginals_df = pd.concat([df_pca, df_fcca])

        filt = [idx for idx in range(marginals_df.shape[0])
                if marginals_df.iloc[idx]['decoder_args']['decoding_window'] == 12]
        marginals_df = marginals_df.iloc[filt]


        # Unpack the epoch from the loader_args
        epochs = [marginals_df.iloc[k]['loader_args']['epoch'] for k in range(marginals_df.shape[0])]
        marginals_df['epoch'] = epochs

    
    elif region in ['VISp']:
        
        marginals_path = root_path + '/decoding_AllenVC_VISp_marginals_glom.pickle'        
        
        with open(marginals_path, 'rb') as f:
            rl = pickle.load(f)
        marginals_df = pd.DataFrame(rl)
        
        unique_loader_args = list({frozenset(d.items()) for d in marginals_df['loader_args']})
        loader_args=dict(unique_loader_args[loader_kwargs[region]['load_idx']])
        marginals_df = apply_df_filters(marginals_df, **{'loader_args':loader_args})
                
        
    return marginals_df    

def get_scalar(df_, stat, neu_idx):
    neu_idx = int(neu_idx)
    if stat == 'decoding_weights':
        if 'decoding_window' in df_.iloc[0]['decoder_params'].keys():
            decoding_win = df_.iloc[0]['decoder_params']['decoding_window']
            c = calc_loadings(df_.iloc[0]['decoding_weights'][2:4].T, d=decoding_win)[neu_idx]
        else:
            c = calc_loadings(df_.iloc[0]['decoding_weights'][2:4].T)[neu_idx]

    elif stat == 'encoding_weights':
        if 'decoding_window' in df_.iloc[0]['decoder_params'].keys():
            decoding_win = df_.iloc[0]['decoder_params']['decoding_window']
            c =  calc_loadings(df_.iloc[0]['encoding_weights'], d=decoding_win)[neu_idx]    
        else:
            c =  calc_loadings(df_.iloc[0]['encoding_weights'])[neu_idx]    

    elif stat in ['su_r2_pos', 'su_r2_vel', 'su_r2_enc', 'su_var', 'su_act', 'su_decoding_r2', 'su_encoding_r2']:
        c = df_.iloc[0][stat][neu_idx]  

    elif stat == 'orientation_tuning':
        c = np.zeros(8)
        for j in range(8):
            c[j] = df_.loc[df_['bin_idx'] == j].iloc[0]['tuning_r2'][j, 2, neu_idx]
        c = np.mean(c)
        # c = odf_.iloc[0]

    return c


def Iscore_prediction_boxplot(r1f_, r1p_, figpath, region):

    # Make a boxplot out of it
    fig, ax = plt.subplots(1, 1, figsize=(1, 5))
    medianprops = dict(linewidth=1, color='b')
    whiskerprops = dict(linewidth=0)

    bplot = ax.boxplot([r1f_, r1p_], patch_artist=True, medianprops=medianprops, notch=False, showfliers=False, whiskerprops=whiskerprops, showcaps=False)
    ax.set_xticklabels(['FBC', 'FFC'], rotation=45)
    if region == 'ML':
        ax.set_ylim([-0.25, 1])
    else:
        ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylabel('Spearman ' + r'$\rho$', fontsize=18)

    colors = ['r', 'k']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    fig_save_path = '%s/iscore_prediction_boxplot_noact_%s.pdf' % (figpath, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)

def Iscore_prediction_histogram(su_r, DIM, figpath, region):

    # Prior to averaging, run tests. Updated for multiple comparisons adjustment. 
    _, p1 = scipy.stats.wilcoxon(su_r[:, 0, 0], su_r[:, 1, 0], alternative='less')
    _, p2 = scipy.stats.wilcoxon(su_r[:, 0, 1], su_r[:, 1, 1], alternative='less')
    _, p3 = scipy.stats.wilcoxon(su_r[:, 0, 2], su_r[:, 1, 2], alternative='less')
    _, p4 = scipy.stats.wilcoxon(su_r[:, 0, 3], su_r[:, 1, 3], alternative='less')
    pvec = np.sort([p1, p3, p4])  
    stats = ['Var', 'Dec', 'Enc']
    porder = np.argsort([p1, p3, p4])

    a1 = pvec[0] * 3
    a2 = pvec[1] * 2
    a3 = pvec[2]

    print('Histogram stats:\n')
    print(f'FBC: Var: {np.mean(su_r[:, 0, 0])}, Dec: {np.mean(su_r[:, 0, 2])}, Enc: {np.mean(su_r[:, 0, 3])}')
    print(f'FFC: Var: {np.mean(su_r[:, 1, 0])}, Dec: {np.mean(su_r[:, 1, 2])}, Enc: {np.mean(su_r[:, 1, 3])}')
    print('p-values:')
    print(', '.join([f'{stats[porder[0]]}:{a1}',
                     f'{stats[porder[1]]}:{a2}',
                     f'{stats[porder[2]]}:{a3}']))

    std_err = np.std(su_r, axis=0).ravel()/np.sqrt(35)
    su_r = np.mean(su_r, axis=0).ravel()
    # Permute so that each statistic is next to each other. No ACT.
    su_r = su_r[[0, 4, 2, 6, 3, 7]]
    std_err = std_err[[0, 4, 2, 6, 3, 7]]

    fig, ax = plt.subplots(figsize=(3, 5),)
    bars = ax.bar([0, 1, 3, 4, 6, 7],
                    su_r,
                    color=['r', 'k', 'r', 'k', 'r', 'k'], alpha=0.65,
                    yerr=std_err, capsize=5)

    # Place numerical values above the bars
    ax.set_ylim([-0.5, 1.1])
    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels(['S.U. Var.', 'Dec. Weights', 'S.U. Enc. ' + r'$r^2$'], rotation=30, fontsize=12, ha='right')

    # Manual creation of legend
    colors = ['r', 'k']
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.65) for c in colors]
    labels = ['FBC', 'FFC']
    ax.legend(handles, labels, loc='lower right', prop={'size': 14})
    ax.set_ylabel('Spearman Correlation ' + r'$\rho$', fontsize=18)
    ax.set_yticks([-0.5, 0, 0.5, 1.])
    ax.tick_params(axis='both', labelsize=16)

    # Horizontal line at 0
    ax.hlines(0, -0.5, 7.5, color='k')
    fig_save_path = '%s/su_spearman_d%d_%s.pdf' % (figpath, DIM, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)


def get_importance_score_predictions(decode_df_all, session_key, data_path, region, DIM):
    # Load dimreduc_df and calculate loadings

    decode_df = decode_df_all

    
    loadings_df = get_loadings_df(decode_df, session_key, DIM)
    su_calcs_df = get_su_calcs(region)
    sessions = np.unique(loadings_df[session_key].values)
    if region in ['VISp']:
        stats = ['su_var', 'su_act', 'decoding_weights', 'su_encoding_r2']
    else:
        stats = ['su_var', 'su_act', 'decoding_weights', 'su_r2_enc']
    ############################ Start Filling in cArray
    carray = []
    for i, session in enumerate(sessions):
            
    
        df_filter = {session_key:sessions[i]}
        df = apply_df_filters(loadings_df, **df_filter)
        carray_ = np.zeros((df.shape[0], len(stats)))
        for j in range(df.shape[0]):                    # Find the corFrelaton between 
            for k, stat in enumerate(stats):
                # Grab the unique identifiers needed
                nidx = df.iloc[j]['nidx']
                try:
                    df_ = apply_df_filters(su_calcs_df, **{session_key:session})
                except:
                    df_ = apply_df_filters(su_calcs_df, session=session)
                carray_[j, k] = get_scalar(df_, stat, nidx)
        carray.append(carray_)


    ############################ Start Filling in su_r
    su_r = np.zeros((len(carray), 2, carray[0].shape[1]))
    keys = ['FCCA_loadings', 'PCA_loadings']
    X, Yf, Yp, x_, yf_, yp_ = [], [], [], [], [], []
    for i in range(len(carray)):
        for j in range(2):
            df_filter = {session_key:sessions[i]}
            df = apply_df_filters(loadings_df, **df_filter)
            x1 = df[keys[j]].values

            if j == 0:
                Yf.extend(x1)
                yf_.append(x1)
            else:
                Yp.extend(x1)
                yp_.append(x1)

            xx = []

            for k in range(carray[0].shape[1]):
                x2 = carray[i][:, k]
                xx.append(x2)
                su_r[i, j, k] = scipy.stats.spearmanr(x1, x2)[0]
        
            xx = np.array(xx).T            
        X.append(xx)
        x_.append(xx)

    X = np.vstack(X)
    Yf = np.array(Yf)[:, np.newaxis]
    Yp = np.array(Yp)[:, np.newaxis]
    assert(X.shape[0] == Yf.shape[0])
    assert(X.shape[0] == Yp.shape[0])

    ############################ Train a linear model to predict loadings from the single unit statistics and then assess the spearman correlation between predicted and actual loadings
    r1p_, r1f_, coefp, coeff, rpcv, rfcv = [], [], [], [], [], []

    for i in range(len(carray)):

        linmodel = LinearRegression().fit(x_[i][:, [0, 2, 3]], np.array(yp_[i])[:, np.newaxis])
        linmodel2 = LinearRegression().fit(x_[i][:, [0, 2, 3]], np.array(yf_[i])[:, np.newaxis])

        yp_pred = linmodel.predict(x_[i][:, [0, 2, 3]])
        yf_pred = linmodel2.predict(x_[i][:, [0, 2, 3]])

        # get normalized coefficients for feature importance assessment
        x__ = StandardScaler().fit_transform(x_[i][:, [0, 2, 3]])
        y__ = StandardScaler().fit_transform(np.array(yp_[i])[:, np.newaxis])

        linmodel = LinearRegression().fit(x__, y__)
        coefp.append(linmodel.coef_.squeeze())

        y__ = StandardScaler().fit_transform(np.array(yf_[i])[:, np.newaxis])
        linmodel = LinearRegression().fit(x__, y__)
        coeff.append(linmodel.coef_.squeeze())

        # Try cross-validation
        rpcv.append(np.mean(cross_val_score(LinearRegression(), x_[i][:, [0, 2, 3]], np.array(yp_[i])[:, np.newaxis], cv=5)))
        rfcv.append(np.mean(cross_val_score(LinearRegression(), x_[i][:, [0, 2, 3]], np.array(yf_[i])[:, np.newaxis], cv=5)))
        r1p_.append(scipy.stats.spearmanr(yp_pred.squeeze(), np.array(yp_[i]).squeeze())[0])
        r1f_.append(scipy.stats.spearmanr(yf_pred.squeeze(), np.array(yf_[i]).squeeze())[0])

    ############################ Run Stats
    stats, p = scipy.stats.wilcoxon(r1p_, r1f_, alternative='greater')
    print(f'S.U. prediction medians:({np.median(r1p_)}, {np.median(r1f_)})')
    print(f'S.U. prediction test:{p}')



    # Get predictions:
    linmodel1 = LinearRegression().fit(X, Yp)
    linmodel2 = LinearRegression().fit(X, np.log10(Yp))

    Yp_pred1 = linmodel1.predict(X)
    Yp_pred2 = linmodel2.predict(X)

    r1p = scipy.stats.spearmanr(Yp_pred1.squeeze(), Yp.squeeze())[0]
    r2p = scipy.stats.spearmanr(Yp_pred2.squeeze(), Yp.squeeze())[0]

    linmodel1 = LinearRegression().fit(X, Yf)
    linmodel2 = LinearRegression().fit(X, np.log10(Yf))

    Yf_pred1 = linmodel1.predict(X)
    Yf_pred2 = linmodel2.predict(X)

    r1f = scipy.stats.spearmanr(Yf_pred1.squeeze(), Yf.squeeze())[0]
    r2f = scipy.stats.spearmanr(Yf_pred2.squeeze(), Yf.squeeze())[0]

    return coefp, coeff, r1f_, r1p_, su_r


def make_Fig5_Iscore_plots(decode_df_all, session_key, data_path, region, DIM, figpath='.'):

    coefp, coeff, r1f_, r1p_, su_r = get_importance_score_predictions(decode_df_all, session_key, data_path, region, DIM)

    # Make Histogram
    Iscore_prediction_boxplot(r1f_, r1p_, figpath, region)
    Iscore_prediction_histogram(su_r, DIM, figpath, region)
    return r1f_, r1p_

'''


MARGINAL PLOTS



'''

def get_marginal_ssa(decoding_df, marginal_df, session_key, region, DIM):

    fold_idcs = np.unique(decoding_df['fold_idx'].values)
    sessions = np.unique(decoding_df[session_key].values)
    dims = np.unique(decoding_df['dim'].values)


    ####################### Find average subspace angles over marginals
    ss_angles = np.zeros((sessions.size, fold_idcs.size, 4, DIM))

    for df_ind, session in tqdm(enumerate(sessions)):
        for fold_ind, fold in enumerate(fold_idcs):
            df_filter = {session_key:session, 'dim':DIM, 'dimreduc_method':'PCA', 'fold_idx':fold}
            dfpca = apply_df_filters(decoding_df, **df_filter)
            assert(dfpca.shape[0] == 1)

            df_filter = {session_key:session, 'dim':DIM, 'dimreduc_method':['LQGCA', 'FCCA'], 'fold_idx':fold}
            dffcca = apply_df_filters(decoding_df, **df_filter)
            assert(dffcca.shape[0] == 1)

            df_filter = {session_key:session, 'dim':DIM, 'dimreduc_method':'PCA', 'fold_idx':fold}
            dfpca_marginal = apply_df_filters(marginal_df, **df_filter)
            assert(dfpca_marginal.shape[0] == 1)

            df_filter = {session_key:session, 'dim':DIM, 'dimreduc_method':['LQGCA', 'FCCA'], 'fold_idx':fold}
            dffcca_marginal = apply_df_filters(marginal_df, **df_filter)
            assert(dffcca_marginal.shape[0] == 1)

            # 0: FBC/FFC
            # 1: FFC/FCCm
            # 2: FBC/FBCm
            # 3: FFBCm/FFCm


            ss_angles[df_ind, fold_ind, 0, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:DIM], dffcca.iloc[0]['coef'])
            ss_angles[df_ind, fold_ind, 1, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:DIM], dfpca_marginal.iloc[0]['coef'][:, 0:DIM])
            ss_angles[df_ind, fold_ind, 2, :] = scipy.linalg.subspace_angles(dffcca.iloc[0]['coef'], dffcca_marginal.iloc[0]['coef'])
            ss_angles[df_ind, fold_ind, 3, :] = scipy.linalg.subspace_angles(dffcca_marginal.iloc[0]['coef'], dfpca_marginal.iloc[0]['coef'][:, 0:DIM])

    return ss_angles


def get_marginal_decoding_differences(decoding_df, marginal_df, session_key, region):

    ####################### Find average decoding differences over marginals
    fold_idcs = np.unique(decoding_df['fold_idx'].values)
    sessions = np.unique(decoding_df[session_key].values)
    dims = np.unique(decoding_df['dim'].values)

    decoding_structs_FFC = np.zeros((sessions.size, fold_idcs.size, dims.size))
    decoding_structs_FBC = np.zeros((sessions.size, fold_idcs.size, dims.size))

    decoding_structs_marginal_FFC = np.zeros((sessions.size, fold_idcs.size, dims.size))
    decoding_structs_marginal_FBC = np.zeros((sessions.size, fold_idcs.size, dims.size))

    for df_ind, session in tqdm(enumerate(sessions)):
        for fold_ind, fold in enumerate(fold_idcs):
            for dim_ind, dim in enumerate(dims):
                df_filter = {session_key:session, 'dim':dim, 'dimreduc_method':'PCA', 'fold_idx':fold}
                df_ = apply_df_filters(decoding_df, **df_filter)
                assert(df_.shape[0] == 1)
                decoding_structs_FFC[df_ind, fold_ind, dim_ind] = get_decoding_performance(df_, region)

                df_filter = {session_key:session, 'dim':dim, 'dimreduc_method':['LQGCA', 'FCCA'], 'fold_idx':fold}
                df_ = apply_df_filters(decoding_df, **df_filter)
                assert(df_.shape[0] == 1)
                decoding_structs_FBC[df_ind, fold_ind, dim_ind] = get_decoding_performance(df_, region)
                
                df_filter = {session_key:session, 'dim':dim, 'dimreduc_method':'PCA', 'fold_idx':fold}
                df_ = apply_df_filters(marginal_df, **df_filter)
                assert(df_.shape[0] == 1)
                decoding_structs_marginal_FFC[df_ind, fold_ind, dim_ind] = get_decoding_performance(df_, region)

                df_filter = {session_key:session, 'dim':dim, 'dimreduc_method':['LQGCA', 'FCCA'], 'fold_idx':fold}
                df_ = apply_df_filters(marginal_df, **df_filter)
                assert(df_.shape[0] == 1)
                decoding_structs_marginal_FBC[df_ind, fold_ind, dim_ind] = get_decoding_performance(df_, region)

    ####################### Average across folds and get deltas
    pca_dec = np.mean(decoding_structs_FFC, axis=1).squeeze()
    fcca_dec = np.mean(decoding_structs_FBC, axis=1).squeeze()

    pca_marginal_dec = np.mean(decoding_structs_marginal_FFC, axis=1).squeeze()
    fcca_marginal_dec = np.mean(decoding_structs_marginal_FBC, axis=1).squeeze()

    pca_dec = pca_dec.reshape(sessions.size, -1)
    fcca_dec = fcca_dec.reshape(sessions.size, -1)
    pca_marginal_dec = pca_marginal_dec.reshape(sessions.size, -1)
    fcca_marginal_dec = fcca_marginal_dec.reshape(sessions.size, -1)

    fcca_delta_marg = fcca_dec - fcca_marginal_dec
    pca_delta_marg = pca_dec - pca_marginal_dec 

    return fcca_delta_marg, pca_delta_marg

def make_Fig5_marginals_plots(decoding_df, session_key, data_path, region, DIM, figpath='.'):

    marginal_df = get_marginal_dfs(region)

    ss_angles = get_marginal_ssa(decoding_df, marginal_df, session_key, region, DIM)
    fcca_delta_marg, pca_delta_marg = get_marginal_decoding_differences(decoding_df, marginal_df, session_key, region)

    dims = np.unique(decoding_df['dim'].values)


    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax = [ax0, ax1]


    ############################ Ax 0: Subspace angles over Marginals ############################
    medianprops = dict(linewidth=1, color='b')
    whiskerprops = dict(linewidth=0)
    bplot = ax[0].boxplot([np.mean(ss_angles[:, :, 2, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 1, :], axis=-1).ravel()], 
                    patch_artist=True, medianprops=medianprops, notch=False, showfliers=False,
                    whiskerprops=whiskerprops, showcaps=False)
    ax[0].set_xticklabels(['FBC/FBCm', 'FFC/FFCm'], rotation=30)
    for label in ax[0].get_xticklabels():
        label.set_horizontalalignment('center')
    ax[0].set_ylim([0, np.pi/2])
    ax[0].set_yticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
    ax[0].set_yticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax[0].tick_params(axis='both', labelsize=16)
    ax[0].set_ylabel('Subspace angles (rads)', fontsize=18)
    colors = ['r', 'k']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)


    ############################ Ax 1: Decoding over Marginals ############################
    sqrt_norm_val = 28
    colors = ['black', 'red', '#781820', '#5563fa']

    dim_lims = {
        'M1_psid': [1, 30],
        'S1_psid': [1, 30],
        'HPC_peanut': [1, 30],
        'VISp': [1, 40]
    }

    dim_ticks = {
        'M1_psid': [1, 10, 20, 30],
        'S1_psid': [1, 10, 20, 30],
        'HPC_peanut': [1, 10, 20, 30],
        'VISp': [1, 20, 40]
    }

    # FCCA and PCADecoding Over Marginals
    ax[1].plot(dims, np.mean(fcca_delta_marg, axis=0), color=colors[1])
    ax[1].plot(dims, np.mean(pca_delta_marg, axis=0), color=colors[0])


    ax[1].fill_between(dims, np.mean(fcca_delta_marg, axis=0) + np.std(fcca_delta_marg, axis=0)/np.sqrt(sqrt_norm_val),
                    np.mean(fcca_delta_marg, axis=0) - np.std(fcca_delta_marg, axis=0)/np.sqrt(sqrt_norm_val), color=colors[1], alpha=0.25, label='__nolegend__')
    ax[1].fill_between(dims, np.mean(pca_delta_marg, axis=0) + np.std(pca_delta_marg, axis=0)/np.sqrt(sqrt_norm_val),
                    np.mean(pca_delta_marg, axis=0) - np.std(pca_delta_marg, axis=0)/np.sqrt(sqrt_norm_val), color=colors[0], alpha=0.25, label='__nolegend__')



    ax[1].set_xlabel('Dimension', fontsize=18)
    # ax[1].set_ylabel(r'$\Delta$' + ' Stim ID Decoding ', fontsize=18, labelpad=-10)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    
    # Start Dimension at 1
    ax[1].set_xlim(dim_lims[region])
    ax[1].set_xticks(dim_ticks[region])


    # Some Statistical Tests
    stat, p1 = scipy.stats.wilcoxon(np.mean(ss_angles[:, :, 2, :], axis=(1, -1)).ravel(), np.mean(ss_angles[:, :, 1, :], axis=(1, -1)).ravel(), alternative='greater')
    stat, p2 = scipy.stats.wilcoxon(np.mean(ss_angles[:, :, 2, :], axis=(1, -1)).ravel(), np.mean(ss_angles[:, :, 0, :], axis=(1, -1)).ravel(), alternative='greater')
    print(f'marginal ssa p vals: FBC vs. FBCm/FFC vs. FFCm: {p1}')


    comp_dim_ind = np.argwhere(dims == DIM)[0][0]
    pkr2f = fcca_delta_marg[:, comp_dim_ind]
    pkr2p = pca_delta_marg[:, comp_dim_ind]

    stat, p = scipy.stats.wilcoxon(fcca_delta_marg[:, comp_dim_ind], pca_delta_marg[:, comp_dim_ind], alternative='greater')
    print('Delta decoding p=%f at d=%d' % (p, dims[comp_dim_ind]))
    #print(np.mean(fcca_delta_marg[:, comp_dim_ind]) - np.mean(pca_delta_marg[:, comp_dim_ind]))

    fig.tight_layout()
    fig_save_path = '%s/decoding_differences_%s.pdf' % (figpath, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)
    return pkr2f, pkr2p, np.mean(ss_angles[:, :, 2, :], axis=(1, -1)).ravel(), np.mean(ss_angles[:, :, 1, :], axis=(1, -1)).ravel()

dim_dict = {
    'M1_psid': 6,
    'M1': 6,
    'S1_psid': 6,
    'S1': 6,
    'HPC_peanut': 11,
    'VISp':10,
}

if __name__ == '__main__':
    regions = ['M1_psid', 'S1_psid', 'HPC_peanut', 'VISp']

    iscore_predf_all = []
    iscore_predp_all = []

    ssaf_all = []
    ssap_all = []

    deltar2f_all = []
    deltar2p_all = []

    for region in tqdm(regions):
        DIM = dim_dict[region]
        figpath = PATH_DICT['figs']

        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        data_path = get_data_path(region)

        # Importance score predictions
        iscore_predf, iscore_predp = make_Fig5_Iscore_plots(df, session_key, data_path, region, DIM, figpath=figpath)
        iscore_predf_all.append(iscore_predf)
        iscore_predp_all.append(iscore_predp)

        pkr2f, pkr2p, ssaf, ssap = make_Fig5_marginals_plots(df, session_key, data_path, region, DIM, figpath=figpath)

        deltar2f_all.append(pkr2f)
        deltar2p_all.append(pkr2p)

        ssaf_all.append(ssaf)
        ssap_all.append(ssap)


        # Pickle away
        with open(PATH_DICT['tmp'] + '/fig5_collated.pkl', 'wb') as f:
            pickle.dump({
                'regions': regions,
                'iscore_predf_all': iscore_predf_all,
                'iscore_predp_all': iscore_predp_all,
                'ssaf_all': ssaf_all,
                'ssap_all': ssap_all,
                'deltar2f_all': deltar2f_all,
                'deltar2p_all': deltar2p_all
            }, f)

    with open(PATH_DICT['tmp'] + '/fig5_collated.pkl', 'rb') as f:
        data = pickle.load(f)
    regions = data['regions']
    iscore_predf_all = data['iscore_predf_all']
    iscore_predp_all = data['iscore_predp_all']
    ssaf_all = data['ssaf_all']
    ssap_all = data['ssap_all']
    deltar2f_all = data['deltar2f_all']
    deltar2p_all = data['deltar2p_all']

    # Test that M1 is stochastically greater than both S1 and HPC for FBC...
    _, p1 = scipy.stats.mannwhitneyu(deltar2f_all[0], deltar2f_all[1], alternative='greater')
    _, p2 = scipy.stats.mannwhitneyu(deltar2f_all[0], deltar2f_all[2], alternative='greater')
    _, p3 = scipy.stats.mannwhitneyu(deltar2f_all[0], deltar2f_all[3], alternative='greater')


    # ..and FFC
    _, p4 = scipy.stats.mannwhitneyu(deltar2p_all[0], deltar2p_all[1], alternative='greater')
    _, p5 = scipy.stats.mannwhitneyu(deltar2p_all[0], deltar2p_all[2], alternative='greater')
    _, p6 = scipy.stats.mannwhitneyu(deltar2p_all[0], deltar2p_all[3], alternative='greater')


    # _, pcorrected1, _, _ = multitest.multipletests([p1, p2], method='holm')
    # _, pcorrected2, _, _ = multitest.multipletests([p3, p4], method='holm')

    # Revised Figure 5 plot - bar plots across areas
    def across_area_barplot(region_data_f, region_data_p, ylim, yticks):

        # Permutation - region order is M1, S1, VISp, HPC
        region_ordering = [0, 1, 3, 2]
        region_data_f = [region_data_f[i] for i in region_ordering]
        region_data_p = [region_data_p[i] for i in region_ordering]
        regions_permuted = [regions[i] for i in region_ordering]

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Calculate medians and IQRs for each region
        medians_f = [np.median(region_data) for region_data in region_data_f]
        medians_p = [np.median(region_data) for region_data in region_data_p]
        
        # Calculate 25th and 75th percentiles for error bars
        q1_f = [np.percentile(region_data, 25) for region_data in region_data_f]
        q3_f = [np.percentile(region_data, 75) for region_data in region_data_f]
        q1_p = [np.percentile(region_data, 25) for region_data in region_data_p]
        q3_p = [np.percentile(region_data, 75) for region_data in region_data_p]
        
        # Calculate error bar lengths
        yerr_f = np.vstack([np.array(medians_f) - np.array(q1_f), np.array(q3_f) - np.array(medians_f)])
        yerr_p = np.vstack([np.array(medians_p) - np.array(q1_p), np.array(q3_p) - np.array(medians_p)])
        
        # Set up bar positions
        x = np.arange(len(regions_permuted))
        width = 0.35
        
        # Create the bar plots
        rects1 = ax.bar(x - width/2, medians_f, width, yerr=yerr_f, label='FBC', color='red', capsize=5, alpha=0.65)
        rects2 = ax.bar(x + width/2, medians_p, width, yerr=yerr_p, label='FFC', color='black', capsize=5, alpha=0.65)
        
        # Add labels, title and legend
        # ax.set_ylabel('Importance Score Prediction (r)')
        # ax.set_xlabel('Brain Region')
        # ax.set_title('Importance Score Predictions by Region')
        ax.set_xticks(x)
        ax.set_xticklabels(regions_permuted)
        # ax.legend()
        
        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=1.0)

        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        
        # Adjust layout
        fig.tight_layout()
        return fig

    fig = across_area_barplot(iscore_predf_all, iscore_predp_all, ylim = [-0.1, 1.1], yticks = [0, 0.5, 1.0])
    fig.savefig(PATH_DICT['figs'] + '/fig5_revised_panel1.pdf', bbox_inches='tight', pad_inches=0)

    fig = across_area_barplot(ssaf_all, ssap_all, ylim = [-0.1 * np.pi/2, np.pi/2 + np.pi/2 * 0.05], yticks = [0, np.pi/8, np.pi/4, 
                                                                                                  3*np.pi/8, np.pi/2])
    fig.savefig(PATH_DICT['figs'] + '/fig5_revised_panel2.pdf', bbox_inches='tight', pad_inches=0)


    # Effect of region?
    stat, p = scipy.stats.kruskal(*iscore_predf_all)
    print(f'kruskal, iscore_predf_all n = {sum([len(x) for x in iscore_predf_all])} p = {p}')
    stat, p = scipy.stats.f_oneway(*iscore_predf_all)
    print(f'f_oneway, iscore_predf_all n = {sum([len(x) for x in iscore_predf_all])} p = {p}')

    stat, p = scipy.stats.kruskal(*iscore_predp_all)
    print(f'kruskal, iscore_predp_all n = {sum([len(x) for x in iscore_predp_all])} p = {p}')
    stat, p = scipy.stats.f_oneway(*iscore_predp_all)
    print(f'f_oneway, iscore_predp_all n = {sum([len(x) for x in iscore_predp_all])} p = {p}')

    stat, p = scipy.stats.kruskal(*ssaf_all)
    print(f'kruskal, ssaf_all n = {sum([len(x) for x in ssaf_all])} p = {p}')
    stat, p = scipy.stats.f_oneway(*ssaf_all)
    print(f'f_oneway, ssaf_all n = {sum([len(x) for x in ssaf_all])} p = {p}')

    stat, p = scipy.stats.kruskal(*ssap_all)
    print(f'kruskal, iscore_predf_all n = {sum([len(x) for x in iscore_predf_all])} p = {p}')
    stat, p = scipy.stats.f_oneway(*ssap_all)
    print(f'f_oneway, ssap_all n = {sum([len(x) for x in ssap_all])} p = {p}')
