import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import pickle
import pandas as pd
from tqdm import tqdm

from dca.cov_util import calc_cross_cov_mats_from_data
import matplotlib.cm as cm
import matplotlib.colors as colors

from region_select import *
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])

from Fig4 import get_loadings_df

hist_ylims = {
    'M1': [0, 500],
    'S1': [0, 105],
    'HPC_peanut':[0, 50],
    'VISp':[0, 125],
}

hist_yticks = {
    'M1': [0, 150, 300],
    'S1': [0, 50, 100],
    'HPC_peanut':[0, 25, 50],
    'VISp': [0, 75, 125],
}

scatter_xlims = {
    'M1': [-0.05, 3.5],
    'S1': [-0.05, 3.5],
    'HPC_peanut':[-0.05, 3.5],
    'VISp': [-0.05, 2],
}

scatter_xticks = {
    'M1': [0, 3.5],
    'S1': [0, 3.5],
    'HPC_peanut': [0, 3.5],
    'VISp': [0, 2],
}

scatter_ylims = {
    'M1': [-0.05, 3.0],
    'S1': [-0.05, 2.0],
    'HPC_peanut':[-0.05, 2.0],
    'VISp': [-0.05, 1.2],
}

scatter_yticks = {
    'M1': [0., 3.0],
    'S1': [0., 2.0],
    'HPC_peanut': [0., 2.0],
    'VISp': [0., 1.0],
}

def calc_psth_su_stats(xall):

    def ccm_thresh(ccm):
        ccm = ccm.squeeze()
        # Normalize
        ccm /= ccm[0]
        thr = 1e-1
        acov_crossing = np.where(ccm < thr)
        if len(acov_crossing[0]) > 0:
            act = np.where(ccm < thr)[0][0]
        else:
            act = len(ccm)

        return act

    n_neurons = len(xall)
    nt = xall[0].shape[1]
    # Stats - dynamic range, autocorrelation time, FFT
    # Consider trial averaged and non-trial averaged variants
    dyn_range = np.zeros((n_neurons, 2))
    act = np.zeros((n_neurons, 2))
    fft = np.zeros((n_neurons, 2))    

    ccmT = int(min(20, xall[0].shape[-1]//2))
    for i in tqdm(range(len(xall))):
        dyn_range[i, 0] = np.max(np.abs(np.mean(xall[i], axis=0)))
        dyn_range[i, 1] = np.mean(np.max(np.abs(xall[i]), axis=1))

        ccm1 = calc_cross_cov_mats_from_data(np.mean(xall[i], axis=0)[:, np.newaxis], ccmT)
        ccm2 = calc_cross_cov_mats_from_data(xall[i][..., np.newaxis], ccmT)
        act[i, 0] = ccm_thresh(ccm1)
        act[i, 1] = ccm_thresh(ccm2)

        # FFT what's the feature? - total power contained beyond the DC component
        N = xall[i].shape[1]
        xfft = scipy.fft.fft(xall[i], axis=1)
        xpsd = np.mean(np.abs(xfft)**2, axis=0)[0:N//2]
        xpsd /= xpsd[0]

        fft[i, 0] = np.sum(xpsd[1:])
        # Trial average and then FFT
        xfft = scipy.fft.fft(np.mean(xall[i], axis=0))
        xpsd = np.abs(xfft**2)[0:N//2]
        xpsd /= xpsd[0]
        fft[i, 1] = np.sum(xpsd[1:])

    return dyn_range, act, fft

def plot_feature_scatter(df, data_path, session_key, region, dim, figpath):

    # Get corresponding loadings
    loadings_df = get_loadings_df(df, session_key, dim=dim)
    sessions = np.unique(loadings_df[session_key].values)
    # Relative FBC/FFC score
    rfbc = np.divide(loadings_df['FCCA_loadings'].values,
                        loadings_df['FCCA_loadings'].values +\
                        loadings_df['PCA_loadings'].values)    
    rffc = np.divide(loadings_df['PCA_loadings'].values,
                        loadings_df['FCCA_loadings'].values +\
                        loadings_df['PCA_loadings'].values)    


    # Load trialized spike rates
    xall = []
    print('Collecting PSTH')
    # Non-trial averaged psth
    for h, session in enumerate(sessions):
        # Do not boxcox        
        load_idx = loader_kwargs[region]['load_idx']  
        unique_loader_args = list({make_hashable(d) for d in df['loader_args']})
        loader_args = dict(unique_loader_args[load_idx])
        loader_args['boxcox'] = None
        x = get_rates_smoothed(data_path, region, session,
                        loader_args=loader_args, 
                        trial_average=False, full_arg_tuple=None)
        xall.extend(x)


    # # Calculate statistics on trialized firing rates
    dyn_range, act, fft = calc_psth_su_stats(xall)
    # Scatter in 3D - color by rfbc        
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap_new = truncate_colormap(cm.RdGy_r, 0., 0.9)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dyn_range[:, 0], act[:, 0], fft[:, 0],
                c=rfbc, edgecolors=(0.6, 0.6, 0.6, 0.6), 
                linewidth=0.01, s=15, cmap=cmap_new)
    ax.set_xlabel('Dynamic Range')
    ax.set_ylabel('Autocorrelation Time')
    ax.set_zlabel('PSD non-DC')
    fig.savefig('psth_scatter_0_%s.pdf' % region)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    ax.scatter(dyn_range[:, 0], fft[:, 0],
                c=rfbc, edgecolors=(0.6, 0.6, 0.6, 0.6), 
                linewidth=0.01, s=15, cmap=cmap_new)

    ax.set_ylim(scatter_ylims[region])
    ax.set_xlim(scatter_xlims[region])

    ax.set_xticks(scatter_xticks[region])
    ax.set_yticks(scatter_yticks[region])
    
    ax.set_xlabel('Peak Amplitude', fontsize=12)
    ax.set_ylabel('Oscillation Strength', fontsize=12)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Rel. FBC Importance', fontsize=12)


    # ax.set_xlabel('Dynamic Range')
    # # ax.set_ylabel('Autocorrelation Time')
    # ax.set_ylabel('PSD non-DC')

    with open('dyn_range_fft.pkl', 'wb') as f:
        pickle.dump(dyn_range, f)
        pickle.dump(fft, f)

    fig.savefig('%s/psth_scatter2D_0_%s.pdf' % (figpath, region),
                 bbox_inches='tight', pad_inches=0)

    return loadings_df, dyn_range, act, fft


DIM_DICT = {
    'M1': 6,
    'S1': 6,
    'HPC_peanut':11,
    'VISp':10,
} 

if __name__ == '__main__':
    
    regions = ['M1', 'S1', 'HPC_peanut', 'VISp']
    for region in regions:

        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        data_path = get_data_path(region)

        loadings_df, dyn_range, act, fft = plot_feature_scatter(df, data_path,  session_key, 
                                                                region,  DIM_DICT[region], 
                                                                PATH_DICT['figs'])        
        print(f"Done with region: {region}")
