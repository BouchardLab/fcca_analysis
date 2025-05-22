import pdb
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy 
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.ndimage import gaussian_filter1d
from dca.methods_comparison import JPCA 
from config import PATH_DICT
from region_select import *
from fig6calcs import get_rates_largs, dim_dict, T_dict
from Fig5 import get_marginal_ssa, get_marginal_dfs, dim_dict as dim_dict_f5

rot_xticks = {
    'M1': [0., 0.15],
    'S1': [0., 0.15],
    'HPC_peanut': [-0.2, -0.1, 0, 0.1, 0.2],
    'ML': [0, 0.25, 0.5],
    'AM': [0, 0.1, 0.3],
    'VISp': [0., 0.15],
    'organoids': [0, 0.1, 0.2, 0.3, 0.4]
    
}

rot_xlims = {
    'M1': [0., 0.15],
    'S1': [0., 0.15],
    'HPC_peanut': [-0.1, 0.2],
    'ML': [0, 0.5],
    'AM': [0, 0.35],
    'VISp': [0., 0.15],
    'organoids': [0, 0.4]
}

dyn_xticks = {
    'M1': [0., 1, 2],
    'S1': [0., 1, 2],
    'HPC_peanut': [0, 1, 2, 3],
    'ML': [0, 1, 2, 3],
    'AM': [0, 1, 2, 3],
    'VISp': [0, 1, 2, 3],
    'organoids': [0., 1, 2]
}

dyn_xlims = {
    'M1': [-0.1, 1.25],
    'S1': [-0.75, 2],
    'HPC_peanut': [0, 2],
    'ML': [0, 3.5],
    'AM': [0, 3],
    'VISp': [0., 3],
    'organoids': [0., 2]
}


def get_random_projections(region, dim, T):
    random_proj_path = PATH_DICT['tmp'] + f'/jpca_tmp_randcontrol_{region}_dim{dim}_T{T}.pkl'
    with open(random_proj_path, 'rb') as f:
        control_results = pickle.load(f)
    controldf = pd.DataFrame(control_results)

    return controldf

def get_df(region, dim, T):
    path = PATH_DICT['tmp'] +f"/jpca_tmp_dimreduc_{region}_dim{dim}_T{T}.pkl"
    with open(path, 'rb') as f:
        results = pickle.load(f)
    df = pd.DataFrame(results)
    return df

def make_rot_plots(region, x, df_fcca, df_pca, jDIM, figpath):


    xpca = x @ df_pca.iloc[0]['coef'][:, 0:jDIM]
    xdca = x @ df_fcca.iloc[0]['coef']

    jpca1 = JPCA(n_components=jDIM, mean_subtract=False)
    jpca1.fit(xpca)

    jpca2 = JPCA(n_components=jDIM, mean_subtract=False)
    jpca2.fit(xdca)

    xpca_j = jpca1.transform(xpca)
    xdca_j = jpca2.transform(xdca)

    xpca_j_mean = np.mean(xpca_j, 0).squeeze()
    xdca_j_mean = np.mean(xdca_j, 0).squeeze()

    ################################### PLOT CODE ##############################

    # Save as two separate figures
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    ax = [ax1, ax2]

    for i in range(0, 25):
        
        trajectory = gaussian_filter1d(xpca_j[i,:,:].squeeze(),  sigma=4, axis=0)

        # Center and normalize trajectories
        trajectory -= trajectory[0]
        #trajectory /= np.linalg.norm(trajectory)

        # Rotate trajectory so that the first 5 timesteps all go off at the same angle
        theta_t = min(15, len(trajectory) - 1)
        theta0 = np.arctan2(trajectory[theta_t, 1], trajectory[theta_t, 0])

        # Rotate *clockwise* by theta
        R = lambda theta: np.array([[np.cos(-1*theta), -np.sin(-theta)], \
                                    [np.sin(-theta), np.cos(theta)]])        
        trajectory = np.array([R(theta0 - np.pi/4) @ t[0:2] for t in trajectory])

        ax[1].plot(trajectory[:, 0], trajectory[:, 1], 'k', alpha=0.5)
        ax[1].arrow(trajectory[-1, 0], trajectory[-1, 1], 
                    trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1], 
                    head_width=0.08, color="k", alpha=0.5)
        
        
        
        
        trajectory = gaussian_filter1d(xdca_j[i,:,:].squeeze(),  sigma=4, axis=0)
        trajectory -= trajectory[0]  # Center trajectories
        #trajectory /= np.linalg.norm(trajectory)

        # Rotate trajectory so that the first 5 timesteps all go off at the same angle
        theta0 = np.arctan2(trajectory[theta_t, 1], trajectory[theta_t, 0])

        trajectory = np.array([R(theta0 - np.pi/4) @ t[0:2] for t in trajectory])

        ax[0].plot(trajectory[:, 0], trajectory[:, 1], '#c73d34', alpha=0.5)
        ax[0].arrow(trajectory[-1, 0], trajectory[-1, 1], 
                    trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1], 
                    head_width=0.05, color="#c73d34", alpha=0.5)


    ax[0].set_aspect('equal')   
    ax[1].set_aspect('equal')   
    # Set xlim and ylim so that all traces fit
    # Get max extent across both plots for consistent axes
    xlims = []
    ylims = []
    for a in ax:
        xlims.extend([l.get_xdata().min() for l in a.lines])
        xlims.extend([l.get_xdata().max() for l in a.lines])
        ylims.extend([l.get_ydata().min() for l in a.lines])
        ylims.extend([l.get_ydata().max() for l in a.lines])
    
    # Add some padding
    xmin, xmax = min(xlims), max(xlims)
    ymin, ymax = min(ylims), max(ylims)
    padding = 0.2 * max(xmax-xmin, ymax-ymin)
    
    # Set same limits for both axes - except HPC
    if region != 'HPC_peanut':
        for a in ax:
            a.set_xlim([xmin-padding, xmax+padding])
            a.set_ylim([ymin-padding, ymax+padding])
    else:
        ax[0].set_xlim([xmin-padding, xmax+padding])
        ax[0].set_ylim([ymin-padding, ymax+padding])
        ax[1].set_xlim([xmin-padding/5, xmax+padding/5])
        ax[1].set_ylim([ymin-padding/5, ymax+padding/5])


    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')
    ax[0].spines['left'].set_position('zero')
    ax[0].spines['bottom'].set_position('zero')
    ax[0].plot(2, 0, ">k", clip_on=False)
    ax[0].plot(0, 2, "^k", clip_on=False)
    ax[0].spines['left'].set_bounds(0, 2)
    ax[0].spines['bottom'].set_bounds(0, 2)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[1].spines['right'].set_color('none')
    ax[1].spines['top'].set_color('none')
    ax[1].spines['left'].set_position('zero')
    ax[1].spines['bottom'].set_position('zero')
    ax[1].spines['left'].set_bounds(0, 2)
    ax[1].spines['bottom'].set_bounds(0, 2)
    ax[1].plot(2, 0, ">k", clip_on=False)
    ax[1].plot(0, 2, "^k", clip_on=False)

    fig1.tight_layout()
    figpath = figpath.replace('.pdf', '_fcca.pdf')
    fig1.savefig(figpath, bbox_inches='tight', pad_inches=0)

    fig2.tight_layout()
    figpath = figpath.replace('_fcca.pdf', '_pca.pdf')
    fig2.savefig(figpath, bbox_inches='tight', pad_inches=0)

def make_traj_amplification_plots(df_, y, DIM):

    dimreduc_method = df_['dimreduc_method'].iloc[0]
    # if region in ['HPC_peanut']:
    #     methods = np.array(['PCA', 'FCCA'])
    # else:
    methods = np.array(['PCA', 'LQGCA'])

    cInd = np.argwhere(methods == dimreduc_method)[0][0]
    colors = ['k', 'r']

    assert(df_.shape[0] == 1)
    V = df_.iloc[0]['coef']
    V = V[:, 0:DIM]        
    # Project data
    yproj = y @ V
    #yproj = np.array([yproj[t0:t0+40] for t0, t1 in dat['transition_times'] if t1 - t0 > 40])
    yproj = np.array([y_ - y_[0] for y_ in yproj])
    dY = np.concatenate(np.diff(yproj, axis=1), axis=0)
    Y_prestate = np.concatenate(yproj[:, :-1], axis=0)

    # Least squares
    A, _, _, _ = np.linalg.lstsq(Y_prestate, dY, rcond=None)
    _, s, _ = np.linalg.svd(A)
    # Iterate the lyapunov equation for 10 timesteps
    P = np.zeros((DIM, DIM))
    for _ in range(10):
        dP = A @ P + P @ A.T + np.eye(DIM)
        P += dP

    eig, U = np.linalg.eig(P)
    # eig, U = np.linalg.eig(scipy.linalg.expm(A.T) @ scipy.linalg.expm(A))
    eig = np.sort(eig)[::-1]
    U = U[:, np.argsort(eig)[::-1]]
    U = U[:, 0:2]
    # Plot smoothed, centered trajectories for all reaches in top 2 dimensions

    # Argsort by the maximum amplitude in the top 2 dimensions
    #trajectory = gaussian_filter1d(yproj, sigma=5, axis=1)
    trajectory = gaussian_filter1d(yproj, sigma=2, axis=1)

    trajectory -= trajectory[:, 0:1, :]
    trajectory = trajectory @ U
    #dyn_range = np.max(np.abs(trajectory), axis=1)
    dyn_range = np.max(np.abs(yproj @ U), axis=1)
    ordering = np.argsort(dyn_range, axis=0)[::-1]

    t0 = trajectory[ordering[:, 0], :, 0]
    t1 = trajectory[ordering[:, 1], :, 1]
    #t0 = trajectory[:, :, 0]
    #t1 = trajectory[:, :, 1]

    f1, a1 = plt.subplots(1, 1, figsize=(4.2, 4))
    f2, a2 = plt.subplots(1, 1, figsize=(4.2, 4))
    ax = [a1, a2]

    for i in range(min(50, t0.shape[0])):

        # Since the sign is arbitrary, let's make sure they all go in the same direction
        # (positive first)
        trace0 = t0[i].copy()
        trace1 = t1[i].copy()

        delta_t0 = trace0[np.argmax(np.abs(trace0))] 
        if delta_t0 < 0:
            trace0 = -trace0
        delta_t1 = trace1[np.argmax(np.abs(trace1))]
        if delta_t1 < 0:
            trace1 = -trace1

        ax[0].plot(np.arange(len(trace0)), trace0, color=colors[cInd], alpha=0.5, linewidth=1.5)
        ax[1].plot(np.arange(len(trace1)), trace1, color=colors[cInd], alpha=0.5, linewidth=1.5)
        #ax[2*j].set_title(np.sum(eig))
        
    for a in ax:
        a.spines['bottom'].set_position('zero')
        # Eliminate upper and right axes
        a.spines['right'].set_color('none')
        a.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        a.xaxis.set_ticks_position('bottom')
        a.yaxis.set_ticks_position('left')

        a.set_xticks([0, len(t0[0]) - 1])
        a.set_xticklabels([])
        a.tick_params(axis='both', labelsize=12)

        a.set_xlabel('Time (s)', fontsize=12)
        a.xaxis.set_label_coords(1.1, 0.56)
        
    # Set y scale according to the current yscale on PCA 0

    if cInd == 0:
        ax[0].set_title('FFC Component 1', fontsize=12)
        ax[1].set_title('FFC Component 2', fontsize=12)
    else:
        ax[0].set_title('FBC Component 1', fontsize=12)
        ax[1].set_title('FBC Component 2', fontsize=12)

    return f1, f2, ax

def make_box_plots(rot_strength_fcca, rot_strength_pca, 
                   dyn_range_fcca, dyn_range_pca, region, figpath, rand_offset=True):

    # Boxplots
    fig0, ax0 = plt.subplots(figsize=(2, 6))

    medianprops = dict(linewidth=1, color='b')
    whiskerprops=dict(linewidth=0)

    # Center relative to random - per recording session
    bplot = ax0.boxplot([rot_strength_fcca, rot_strength_pca], patch_artist=True, 
                    medianprops=medianprops, notch=False, vert=True, showfliers=False, 
                    widths=[0.3, 0.3],
                    whiskerprops=whiskerprops, showcaps=False)
    
    _, p1 = scipy.stats.wilcoxon(rot_strength_fcca, rot_strength_pca, alternative='greater')
    _, p2 = scipy.stats.wilcoxon(rot_strength_fcca, rot_strength_pca, alternative='less')
    print('Rotational test p: %f' % min(p1, p2))

    # test that each is stochastically greater than the median random
    _, p1 = scipy.stats.wilcoxon(rot_strength_fcca, alternative='greater')
    _, p2 = scipy.stats.wilcoxon(rot_strength_pca, alternative='greater')


    method1 = 'FBC'
    method2 = 'FFC'

    ax0.set_xticklabels([], fontsize=12)
    #ax0.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    #ax0.set_xlim([0, 0.2])
    ax0.tick_params(axis='both', labelsize=12)
    #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
    ax0.set_ylabel('', fontsize=9)
    #ax.set_title('****', fontsize=14)

    # ax0.invert_xaxis()
    # ax0.set_yticks(rot_xticks[region])
    # ax0.set_yticklabels([])
    # ax0.set_ylim(rot_xlims[region])
    colors = ['red', 'black', 'blue']   
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    fig0.tight_layout()
    if not rand_offset:
        save_fig_path = '%s/rot_strength_box_plots_%s_rand_offset_false.pdf' % (figpath, region)
    else:
        save_fig_path = '%s/rot_strength_box_plots_%s.pdf' % (figpath, region)
    fig0.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)


    fig1, ax1 = plt.subplots(figsize=(2, 6))

    # fill with colors

    whiskerprops = dict(linewidth=0)

    bplot = ax1.boxplot([dyn_range_fcca, dyn_range_pca], patch_artist=True, 
                    medianprops=medianprops, notch=False, vert=True, showfliers=False, widths=[0.3, 0.3],
                    whiskerprops=whiskerprops, showcaps=False)

    _, p1 = scipy.stats.wilcoxon(dyn_range_fcca, alternative='greater')
    _, p2 = scipy.stats.wilcoxon(dyn_range_pca, alternative='greater')
    _, p3 = scipy.stats.wilcoxon(dyn_range_fcca, dyn_range_pca, alternative='less')
    _, p3_2 = scipy.stats.wilcoxon(dyn_range_fcca, dyn_range_pca, alternative='greater')
    print('Amplification p test: %f' % min(p3, p3_2))
    method1 = 'FBC'
    method2 = 'FFC'

    # ax1.set_xticklabels([], fontsize=12)
    # ax1.set_yticklabels([], fontsize=12)
    # ax1.set_yticks(dyn_xticks[region])
    # ax1.set_ylim(dyn_xlims[region])
    ax1.tick_params(axis='both', labelsize=12)
    #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
    ax1.set_ylabel('', fontsize=9)
    #ax.set_title('****', fontsize=14)

    # ax1.invert_xaxis()

    # fill with colors
    colors = ['red', 'black', 'blue']   
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    fig1.tight_layout()
    if not rand_offset:
        save_fig_path = '%s/dyn_rng_box_plots_%s_rand_offset_false.pdf' % (figpath, region)
    else:
        save_fig_path = '%s/dyn_rng_box_plots_%s.pdf' % (figpath, region)
    fig1.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)

def collate_jpca_results(decoding_df, session_key, region, DIM, T=None):

    if T is None:
        T = T_dict[region]

    ################## Load random projections for later comparison
    controldf = get_random_projections(region, DIM, T)
    A_df = get_df(region, DIM, T)

    sessions = np.unique(decoding_df[session_key].values)

    rot_strength = np.zeros((len(sessions), 2))
    rot_strength_maxnorm = np.zeros((len(sessions), 2))
    rot_strength_stdnorm = np.zeros((len(sessions), 2))
    jDIM = DIM - 1 if DIM % 2 != 0 else DIM
    raw_jpca_eigvals = np.zeros((len(sessions), 2, jDIM))
    dyn_range = np.zeros((len(sessions), 2))
    dyn_range_maxnorm = np.zeros((len(sessions), 2))
    dyn_range_stdnorm = np.zeros((len(sessions), 2))
    dyn_ranges = np.zeros((len(sessions), 2, DIM))
    dyn_ranges_maxnorm = np.zeros((len(sessions), 2, DIM))
    dyn_ranges_stdnorm = np.zeros((len(sessions), 2, DIM))

    control_reps = len(np.unique(controldf['inner_rep'].values))
    rot_strength_control = np.zeros((len(sessions), control_reps))
    rot_strength_maxnorm_control = np.zeros((len(sessions), control_reps))
    rot_strength_stdnorm_control = np.zeros((len(sessions), control_reps))
    dyn_range_control = np.zeros((len(sessions), control_reps))
    dyn_range_maxnorm_control = np.zeros((len(sessions), control_reps))
    dyn_range_stdnorm_control = np.zeros((len(sessions), control_reps))

    for i in range(len(sessions)):
        for j, dimreduc_method in enumerate(['LQGCA', 'PCA']):
            df_ = apply_df_filters(A_df, **{session_key: sessions[i], 'dimreduc_method':dimreduc_method})
            eigs = df_.iloc[0]['jeig']
            rot_strength[i, j] = np.sum(np.abs(eigs))/2
            dyn_range[i, j] = df_.iloc[0]['dyn_range']
            raw_jpca_eigvals[i, j, :] = np.abs(eigs)    
            # Normalized versions
            eigs = df_.iloc[0]['jeig_maxnorm']
            rot_strength_maxnorm[i, j] = np.sum(np.abs(eigs))/2
            eigs = df_.iloc[0]['jeig_stdnorm']
            rot_strength_stdnorm[i, j] = np.sum(np.abs(eigs))/2

            dyn_range_maxnorm[i, j] = df_.iloc[0]['dyn_range_maxnorm']
            dyn_range_stdnorm[i, j] = df_.iloc[0]['dyn_range_stdnorm']

        for j in range(control_reps):
            df_ = apply_df_filters(controldf, **{session_key: sessions[i], 'inner_rep':j})
            assert(df_.shape[0] == 1)

            eigs = df_.iloc[0]['jeig']
            rot_strength_control[i, j] = np.sum(np.abs(eigs))/2
            dyn_range_control[i, j] = df_.iloc[0]['dyn_range']

            eigs = df_.iloc[0]['jeig_maxnorm']
            rot_strength_maxnorm_control[i, j] = np.sum(np.abs(eigs))/2
            eigs = df_.iloc[0]['jeig_stdnorm']
            rot_strength_stdnorm_control[i, j] = np.sum(np.abs(eigs))/2

            dyn_range_maxnorm_control[i, j] = df_.iloc[0]['dyn_range_maxnorm']
            dyn_range_stdnorm_control[i, j] = df_.iloc[0]['dyn_range_stdnorm']


    mu_rot_control = np.mean(rot_strength_control, axis=1)
    rot_strength_fcca = rot_strength[:, 0]
    rot_strength_pca = rot_strength[:, 1]

    mu_rot_maxnorm_control = np.mean(rot_strength_maxnorm_control, axis=1)
    rot_strength_maxnorm_fcca = rot_strength_maxnorm[:, 0]
    rot_strength_maxnorm_pca = rot_strength_maxnorm[:, 1]

    mu_rot_stdnorm_control = np.mean(rot_strength_stdnorm_control, axis=1)
    rot_strength_stdnorm_fcca = rot_strength_stdnorm[:, 0]
    rot_strength_stdnorm_pca = rot_strength_stdnorm[:, 1]

    mu_dyn_control = np.mean(dyn_range_control, axis=1)
    dyn_range_fcca = dyn_range[:, 0]
    dyn_range_pca = dyn_range[:, 1]

    mu_dyn_maxnorm_control = np.mean(dyn_range_maxnorm_control, axis=1)
    dyn_range_maxnorm_fcca = dyn_range_maxnorm[:, 0]
    dyn_range_maxnorm_pca = dyn_range_maxnorm[:, 1]

    mu_dyn_stdnorm_control = np.mean(dyn_range_stdnorm_control, axis=1)
    dyn_range_stdnorm_fcca = dyn_range_stdnorm[:, 0]
    dyn_range_stdnorm_pca = dyn_range_stdnorm[:, 1]

    return {
        'rot_strength_fcca': rot_strength_fcca,
        'rot_strength_pca': rot_strength_pca,
        'raw_jpca_eigvals_fcca': raw_jpca_eigvals[:, 0, :],
        'raw_jpca_eigvals_pca': raw_jpca_eigvals[:, 1, :],
        'dyn_range_fcca': dyn_range_fcca,
        'dyn_range_pca': dyn_range_pca,
        'rot_strength_maxnorm_fcca': rot_strength_maxnorm_fcca,
        'rot_strength_maxnorm_pca': rot_strength_maxnorm_pca,
        'dyn_range_maxnorm_fcca': dyn_range_maxnorm_fcca,
        'dyn_range_maxnorm_pca': dyn_range_maxnorm_pca,
        'rot_strength_stdnorm_fcca': rot_strength_stdnorm_fcca,
        'rot_strength_stdnorm_pca': rot_strength_stdnorm_pca,
        'dyn_range_stdnorm_fcca': dyn_range_stdnorm_fcca,
        'dyn_range_stdnorm_pca': dyn_range_stdnorm_pca,
        'mu_rot_control': mu_rot_control,
        'mu_dyn_control': mu_dyn_control,
        'mu_rot_maxnorm_control': mu_rot_maxnorm_control,
        'mu_dyn_maxnorm_control': mu_dyn_maxnorm_control,
        'mu_rot_stdnorm_control': mu_rot_stdnorm_control,
        'mu_dyn_stdnorm_control': mu_dyn_stdnorm_control,
    }

def collate_fig6justification(regions, dim_set, recalculate=False):    
    dim_dicts = [
        {
            'M1': 6,
            'S1': 6,
            'HPC_peanut': 11,
            'VISp': 10
        },
        {
            'M1': 6,
            'S1': 6,
            'HPC_peanut': 6,
            'VISp': 6
        }
    ]
    if not os.path.exists(PATH_DICT['tmp'] + f'/fig6_justification_dimset{dim_set}.pkl'):
        recalculate = True

    if recalculate:
        ss_angles_all = []
        rot_dyn_results_dicts = []
        dims = []
        for region in tqdm(regions):
            df, session_key = load_decoding_df(region, **loader_kwargs[region])
            data_path = get_data_path(region)

            # Get the marginal subspace angles 
            marginal_df = get_marginal_dfs(region)

            # ss_angles: (sessions, folds, 4, dim)
            # 0: FBC/FFC
            # 1: FFC/FCCm
            # 2: FBC/FBCm
            # 3: FFBCm/FFCm
            ss_angles = get_marginal_ssa(df, marginal_df, session_key, region, dim_dict_f5[region])
            ss_angles_all.append(ss_angles)

            # Also load decodingvdim results and save the dimension at which the difference 
            # in FBC/FFC decoding accuracy is maximized
            with open(PATH_DICT['tmp'] + '/decodingvdim_across_regions.pkl', 'rb') as f:
                r2f_across_regions = pickle.load(f)
                r2p_across_regions = pickle.load(f)
                r2_sup_across_regions = pickle.load(f)
                regions_decodingvdim = pickle.load(f)

            # Order by how we have specified regions
            dim_delta_max = []
            for i in range(len(regions)):
                index = regions_decodingvdim.index(regions[i])
                dr2 = r2f_across_regions[index] - r2p_across_regions[index]
                dim_delta_max.append(np.argmax(dr2, axis=1) + 1)

            # For compatibility with fig6 calculations, we have to remove the _psid suffix
            if '_psid' in region:
                region = region.split('_psid')[0]

            DIM = dim_dicts[dim_set][region]
            dims.append(DIM)
            results_dict = collate_jpca_results(df, session_key, region, DIM)
            rot_dyn_results_dicts.append(results_dict)
        with open(PATH_DICT['tmp'] + f'/fig6_justification_dimset{dim_set}.pkl', 'wb') as f:
            pickle.dump({'ss_angles_all':ss_angles_all, 
                            'rot_dyn_results_dicts': rot_dyn_results_dicts,
                            'dims': dims,
                            'dim_delta_max': dim_delta_max}, f)
    else:
        with open(PATH_DICT['tmp'] + f'/fig6_justification_dimset{dim_set}.pkl', 'rb') as f:
            data = pickle.load(f)

        ss_angles_all = data['ss_angles_all']
        rot_dyn_results_dicts = data['rot_dyn_results_dicts']
        dims = data['dims']
        dim_delta_max = data['dim_delta_max']

    return ss_angles_all, rot_dyn_results_dicts, dims, dim_delta_max

def plot_Fig6(decoding_df, session_key, data_path, region, DIM, figpath='.', rand_offset=True):

    jDIM = DIM - 1 if DIM % 2 != 0 else DIM    # jPCA dimension must be even

    results_dict = collate_jpca_results(decoding_df, session_key, region, DIM)

    rot_strength_fcca = results_dict['rot_strength_fcca']
    rot_strength_pca = results_dict['rot_strength_pca']
    dyn_range_fcca = results_dict['dyn_range_fcca']
    dyn_range_pca = results_dict['dyn_range_pca']

    # Offset to random
    if rand_offset:
        rot_strength_fcca = rot_strength_fcca - results_dict['mu_rot_control']
        rot_strength_pca = rot_strength_pca - results_dict['mu_rot_control']
        dyn_range_fcca = dyn_range_fcca - results_dict['mu_dyn_control']
        dyn_range_pca = dyn_range_pca - results_dict['mu_dyn_control']

    make_box_plots(rot_strength_fcca, rot_strength_pca, 
                   dyn_range_fcca, dyn_range_pca, region, figpath, rand_offset)

    # Also plot the raw jpCA eigvals
    fig, ax = plt.subplots(figsize=(4, 4))
    eig_f = results_dict['raw_jpca_eigvals_fcca']
    eig_p = results_dict['raw_jpca_eigvals_pca']

    for j in range(eig_f.shape[1]):
        ax.scatter(2 * np.ones(eig_f.shape[0]) * j, eig_f[:, j], color='red', alpha=0.5)
        ax.scatter(2 * np.ones(eig_p.shape[0]) * j + 0.5, eig_p[:, j], color='black', alpha=0.5)
        # Connect the scatter points by a line
        ax.plot(np.hstack([2 * np.ones((eig_f.shape[0], 1)) * j, 2 * np.ones((eig_p.shape[0], 1)) * j + 0.5]), 
                np.hstack([eig_f[:, j][:, np.newaxis], eig_p[:, j][:, np.newaxis]]), color='gray', alpha=0.5)

    ax.set_xticks(np.arange(1, eig_f.shape[1] + 1))
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('jPCA Eigenvalues')
    # ax.legend()
    fig.tight_layout()
    fig.savefig(figpath + f'/raw_jpca_eigvals_{region}.pdf', bbox_inches='tight', pad_inches=0)
    # # Plot projections for the sessions
    # figdir = f'{figpath}/{region}_fig6traces'
    # if not os.path.exists(figdir):
    #     os.makedirs(figdir)

    # sessions = np.unique(decoding_df[session_key].values)
    # for i, session in enumerate(sessions):
    #     if region in ['AM', 'ML']:
    #         fcca_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 
    #                     'dimreduc_method':'LQGCA', 'loader_args':{'region': region}}
    #         pca_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 
    #                     'dimreduc_method':'PCA', 'loader_args':{'region': region}}
    #     else:
    #         fcca_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 'dimreduc_method':'LQGCA'}
    #         pca_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 'dimreduc_method':'PCA'}

    #     df_fcca = apply_df_filters(decoding_df, **fcca_filter)
    #     df_pca = apply_df_filters(decoding_df, **pca_filter)

    #     y = get_rates_largs(T_dict[region], decoding_df, data_path, region, session)
    #     figpath = f'{figdir}/rot_{session}.pdf'     
    #     make_rot_plots(region, y, df_fcca, df_pca, jDIM, figpath)

    #     figpath_fcca = f'{figdir}/ampl_{session}_fcca.pdf'
    #     f1_fcca, f2_fcca, ax_fcca = make_traj_amplification_plots(df_fcca, y, jDIM)


    #     figpath_pca = f'{figdir}/ampl_{session}_pca.pdf'
    #     f1_pca, f2_pca, ax_pca = make_traj_amplification_plots(df_pca, y, jDIM)

    #     # Standardize y axes
    #     ylims = []
    #     for ax in [ax_fcca, ax_pca]:
    #         for a in ax:
    #             ylims.extend([l.get_ydata().min() for l in a.lines])
    #             ylims.extend([l.get_ydata().max() for l in a.lines])
        
    #     # Add some padding
    #     ymax = max(np.abs(np.min(ylims)), np.max(ylims))
    #     ymin = -ymax
    #     padding = 0.2 * (ymax - ymin)
        
    #     # Set same y limits for all axes
    #     for ax in [ax_fcca, ax_pca]:

    #         if region == 'HPC_peanut':
    #             for a in ax:
    #                 a.set_ylim([-2, 3])
    #         elif region == 'M1':
    #             for a in ax:
    #                 a.set_ylim([-2, 8])
    #         else:
    #             for a in ax:
    #                 a.set_ylim([ymin-padding, ymax+padding])
    #                 # Set the ticks to the nearest integer below the ymin and ymax 
    #                 a.set_yticks([np.ceil(ymin - padding), np.floor(ymax + padding)])


    #     figpath_fcca_a = figpath_fcca.replace('.pdf', '_a.pdf') 
    #     figpath_fcca_b = figpath_fcca.replace('.pdf', '_b.pdf')
    #     figpath_pca_a = figpath_pca.replace('.pdf', '_a.pdf')
    #     figpath_pca_b = figpath_pca.replace('.pdf', '_b.pdf')


    #     f1_fcca.savefig(figpath_fcca_a, bbox_inches='tight', pad_inches=0)
    #     f2_fcca.savefig(figpath_fcca_b, bbox_inches='tight', pad_inches=0)
    #     f1_pca.savefig(figpath_pca_a, bbox_inches='tight', pad_inches=0)
    #     f2_pca.savefig(figpath_pca_b, bbox_inches='tight', pad_inches=0)

    #     # Close all figures
    #     plt.close('all')

    return rot_strength_fcca, rot_strength_pca, dyn_range_fcca, dyn_range_pca

def plot_Fig6justification(regions, ss_angles_all, rot_dyn_results_dicts, dims, dim_delta_max, dual_axes=False):

    # FBC/FBCm and FFC/FFCm - average over folds and angles
    fbc_fbcm = [s[:, :, 2, :].mean(axis=(1, 2)) for s in ss_angles_all]
    ffc_ffcm = [s[:, :, 1, :].mean(axis=(1, 2)) for s in ss_angles_all]

    # Panel 1 - FBC/FBCm vs. delta rot strength, rand offset true
    fig, ax = plt.subplots(figsize=(4, 4))
    delta_rot = [rot_dyn_results_dicts[i]['rot_strength_fcca'] - rot_dyn_results_dicts[i]['rot_strength_pca'] \
                 for i in range(len(regions))]

    region_names = ['M1', 'S1', 'HPC', 'VISp']
    region_markers = ['o', 'p', 'v', 's']

    region_colors_diff = ['purple', 'purple', 'g', '#bf842c']

    for i in range(len(regions)):   

        # Calculate means and standard errors for each region
        fbc_fbcm_mean = np.mean(fbc_fbcm[i])
        fbc_fbcm_std = np.std(fbc_fbcm[i])         
        fbc_fbcm_se = fbc_fbcm_std / np.sqrt(len(fbc_fbcm[i]))
        
        ffc_ffcm_mean = np.mean(ffc_ffcm[i])
        ffc_ffcm_std = np.std(ffc_ffcm[i])         
        ffc_ffcm_se = ffc_ffcm_std / np.sqrt(len(ffc_ffcm[i]))
        
        delta_rot_mean = np.mean(delta_rot[i])
        delta_rot_std = np.std(delta_rot[i])         
        delta_rot_se = delta_rot_std / np.sqrt(len(delta_rot[i]))

        if dual_axes:
            # ax.scatter(fbc_fbcm[i], delta_rot[i], marker=region_markers[i], 
            #         label=region_names[i], facecolor='r',
            #         alpha=0.35, edgecolor='none')
            # ax.scatter(ffc_ffcm[i], delta_rot[i], marker=region_markers[i], 
            #         label=region_names[i], facecolor='k',
            #         alpha=0.35, edgecolor='none')


            ax.errorbar(fbc_fbcm_mean, delta_rot_mean, 
                      xerr=fbc_fbcm_std, yerr=delta_rot_std,
                      marker=region_markers[i], label=region_names[i], 
                      mec='r', capsize=3, mfc=(1., 0, 0, 0.5),
                      ecolor=(1., 0, 0, 0.5))
            
            # Plot mean with error bars for FFC/FFCm
            ax.errorbar(ffc_ffcm_mean, delta_rot_mean,
                      xerr=ffc_ffcm_std, yerr=delta_rot_std,
                      marker=region_markers[i], label=region_names[i],
                      mec='k', capsize=3, mfc=(0., 0., 0., 0.5),
                      ecolor=(0., 0., 0., 0.5))


        else:
            ax.scatter(fbc_fbcm[i], delta_rot[i], marker=region_markers[i], 
                    label=region_names[i], color=region_colors_diff[i],
                    alpha=0.5)

    if dual_axes:
        ax.set_xlabel('null')
    else:
        ax.set_xlabel('FBC/FBCm')
    ax.set_xlim([0, np.pi/2])
    ax.set_xticks([0, np.pi/4, np.pi/2])
    ax.set_xticklabels([])
    ax.set_ylabel('dRot')
    ax.set_ylim([-0.325, 0.125])
    ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1])

    # Horizontal dashed line at 0
    ax.axhline(0, 0, np.pi/2, color='gray', linestyle='--')

    # Print out pearson correlation and LDA clssification accuracy
    x_ = []
    for angles in fbc_fbcm:
        x_.extend(angles)
    y_ = []
    for rot in delta_rot:
        y_.extend(rot)

    # Plot results of linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_, y_)
    ax.plot(np.linspace(0, np.pi/2, 100), slope * np.linspace(0, np.pi/2, 100) + intercept, color='red')
    print('Pearson r FBC/FBCm: ', scipy.stats.pearsonr(x_, y_))
    print('Spearman r FBC/FBCm: ', scipy.stats.spearmanr(x_, y_))

    x_ = []
    for angles in ffc_ffcm:
        x_.extend(angles)
    y_ = []
    for rot in delta_rot:
        y_.extend(rot)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_, y_)
    ax.plot(np.linspace(0, np.pi/2, 100), slope * np.linspace(0, np.pi/2, 100) + intercept, color='black')
    print('Pearson r FFC/FFCm: ', scipy.stats.pearsonr(x_, y_))
    print('Spearman r FFC/FFCm: ', scipy.stats.spearmanr(x_, y_))
    fig.tight_layout()
    fig.savefig(PATH_DICT['figs'] + '/fig6_justification_panel1.pdf', 
                bbox_inches='tight', pad_inches=0)

    # Panel 2 - FFC/FFCm vs. delta dyn strength, rand offset true
    fig, ax = plt.subplots(figsize=(4, 4))

    delta_dyn = [rot_dyn_results_dicts[i]['dyn_range_fcca'] - rot_dyn_results_dicts[i]['dyn_range_pca'] \
                 for i in range(len(regions))]

    # Region elipses
    def region_elipse(points, **plotting_kwargs):
        mean = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        
        # Get eigenvalues and eigenvectors of the covariance matrix
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Calculate the angle of the ellipse
        theta = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        
        # Calculate the width and height of the ellipse (2 standard deviations)
        width, height = 2 * np.sqrt(eigenvals)
        
        # Create the ellipse
        ellipse = Ellipse(xy=mean, width=width, height=height, 
                            angle=np.degrees(theta), **plotting_kwargs)
        return ellipse

    for i in range(len(regions)):   
        # Calculate means and standard errors for each region
        fbc_fbcm_mean = np.mean(fbc_fbcm[i])
        fbc_fbcm_std = np.std(fbc_fbcm[i])         
        fbc_fbcm_se = fbc_fbcm_std / np.sqrt(len(fbc_fbcm[i]))
        
        ffc_ffcm_mean = np.mean(ffc_ffcm[i])
        ffc_ffcm_std = np.std(ffc_ffcm[i])         
        ffc_ffcm_se = ffc_ffcm_std / np.sqrt(len(ffc_ffcm[i]))
        
        delta_dyn_mean = np.mean(delta_dyn[i])
        delta_dyn_std = np.std(delta_dyn[i])         
        delta_dyn_se = delta_dyn_std / np.sqrt(len(delta_dyn[i]))

        if dual_axes:

            # Mean and error bars per region only
            ax.errorbar(fbc_fbcm_mean, delta_dyn_mean, 
                      xerr=fbc_fbcm_std, yerr=delta_dyn_std,
                      marker=region_markers[i], label=region_names[i], 
                      mec='r', capsize=3, mfc=(1., 0, 0, 0.5),
                      ecolor=(1., 0, 0, 0.5))
            
            # Plot mean with error bars for FFC/FFCm
            ax.errorbar(ffc_ffcm_mean, delta_dyn_mean,
                      xerr=ffc_ffcm_std, yerr=delta_dyn_std,
                      marker=region_markers[i], label=region_names[i],
                      mec='k', capsize=3, mfc=(0., 0., 0., 0.5),
                      ecolor=(0., 0., 0., 0.5))


            # ax.scatter(ffc_ffcm[i], delta_dyn[i], marker=region_markers[i], 
            #             label=region_names[i], facecolor='k',
            #             alpha=0.3, edgecolor='none')

            # # Add an elipse that encompasses the mean and standard deviation along each axis
            # # Calculate mean and covariance of the points
            # points = np.column_stack((ffc_ffcm[i], delta_dyn[i]))
            # ellipse = region_elipse(points, edgecolor='k', alpha=1.0, facecolor='none', linestyle='-')
            # # Add the ellipse to the plot
            # ax.add_patch(ellipse)
            # ax.scatter(fbc_fbcm[i], delta_dyn[i], marker=region_markers[i], 
            #             label=region_names[i], facecolor='r',
            #             alpha=0.3, edgecolor='none')
            # points = np.column_stack((fbc_fbcm[i], delta_dyn[i]))
            # ellipse = region_elipse(points, edgecolor='r', alpha=1.0, facecolor='none', linestyle='-')
            # # Add the ellipse to the plot
            # ax.add_patch(ellipse)
        else:
            ax.scatter(ffc_ffcm[i], delta_dyn[i], marker=region_markers[i], 
                        label=region_names[i], color=region_colors_diff[i],
                        alpha=0.5)

    if dual_axes:
        ax.set_xlabel('null')
    else:
        ax.set_xlabel('FFC/FFCm')
 
    ax.set_xlim([0, np.pi/2])
    ax.set_xticks([0, np.pi/4, np.pi/2])
    ax.set_xticklabels([])
    ax.set_ylabel('dDyn')
    ax.set_ylim([-1.5, 1.5])
    ax.set_yticks([-1.5, 0, 1.5])

    # Horizontal dashed line at 0
    ax.axhline(0, 0, np.pi/2, color='gray', linestyle='--')

    fig.tight_layout()
    fig.savefig(PATH_DICT['figs'] + '/fig6_justification_panel2.pdf',
                 bbox_inches='tight', pad_inches=0)


    # 3 classes - M1/S1, HPC, VISp
    labels = []
    labels.extend([0] * len(fbc_fbcm[0]))
    labels.extend([0] * len(fbc_fbcm[1]))
    labels.extend([1] * len(fbc_fbcm[2]))
    labels.extend([2] * len(fbc_fbcm[3]))
    labels = np.array(labels)
    features = []
    for i in range(len(regions)):
        features.extend(np.hstack([ffc_ffcm[i][:, np.newaxis], 
                                   delta_dyn[i][:, np.newaxis]]))
    features = np.array(features)
    # LDA classification accuracy   
    lda = LinearDiscriminantAnalysis()

    lda.fit(features, labels.reshape(-1, 1))
    print('LDA classification accuracy: ', lda.score(features, labels.reshape(-1, 1)))

if __name__ == '__main__':

    #regions = ['M1', 'S1', 'M1_maze', 'HPC_peanut', 'AM', 'ML']
    regions = ['M1', 'S1', 'HPC_peanut', 'VISp']
    dim_set = 0
    ss_angles_all, rot_dyn_results_dicts, dims, dim_delta_max = \
        collate_fig6justification(regions, dim_set)

    plot_Fig6justification(regions, ss_angles_all, rot_dyn_results_dicts, dims, dim_delta_max, dual_axes=True)
    sys.exit()
    regions = ['M1', 'S1', 'HPC_peanut', 'VISp']

    dyn_range_allf = []
    dyn_range_allp = []
    rot_allf = []
    rot_allp = []

    for region in tqdm(regions):
        print(f"Starting region: {region}")

        figpath = PATH_DICT['figs']
        DIM = dim_dict[region]
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        data_path = get_data_path(region) 
        rot_f, rot_p, dr_f, dr_p = plot_Fig6(df, session_key, 
                                            data_path, region, DIM, figpath=figpath, rand_offset=False)

        rot_allf.append(rot_f)
        rot_allp.append(rot_p)

        dyn_range_allf.append(dr_f)
        dyn_range_allp.append(dr_p)

        print(f"Done with region: {region}. Region stats: ")
    
    # Across area statistical tests    
    # _, p1 = scipy.stats.mannwhitneyu(dyn_range_allf[0], dyn_range_allf[1], alternative='greater')
    # _, p2 = scipy.stats.mannwhitneyu(dyn_range_allp[0], dyn_range_allp[1], alternative='greater')

    # _, p3 = scipy.stats.mannwhitneyu(rot_allf[0], rot_allf[1], alternative='greater')
    # _, p4 = scipy.stats.mannwhitneyu(rot_allp[0], rot_allp[1], alternative='greater')

