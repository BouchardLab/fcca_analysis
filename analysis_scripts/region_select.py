import pickle
import numpy as np
import pandas as pd
import pdb
import itertools
import sys
import os
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])

from loaders import (load_sabes, load_peanut, reach_segment_sabes,
                      segment_peanut, load_AllenVC)
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

loader_kwargs = {
    'M1': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'M1_psid': {'load_idx':0, 'dec_idx':0, 'dr_idx':0, 'use_highd':False},
    'S1': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'S1_psid': {'load_idx':0, 'dec_idx':0, 'dr_idx':0, 'use_highd':False},
    'HPC_peanut': {'load_idx':0, 'dec_idx':0, 'dr_idx':0}, 
    'VISp':{'load_idx':0, 'dec_idx':0, 'dr_idx':0},
}



def filter_by_dict(df, root_key, dict_filter):

    col = df[root_key].values

    filtered_idxs = []

    for i, c in enumerate(col):
        match = True
        for key, val in dict_filter.items():
            if key in c.keys():
                if c[key] != val:
                    match = False
            else:
                match = False
        if match:
            filtered_idxs.append(i)

    return df.iloc[filtered_idxs]

# Shortcut to apply multiple filters to pandas dataframe
def apply_df_filters(dtfrm, invert=False, reset_index=True, **kwargs):

    filtered_df = dtfrm

    for key, value in kwargs.items():

        # If the value is the dict
        if type(value) == dict:
            filtered_df = filter_by_dict(filtered_df, key, value)
        else:
            if type(value) == list:
                matching_idxs = []
                for v in value:
                    df_ = apply_df_filters(filtered_df, reset_index=False, **{key:v})
                    if invert:
                        matching_idxs.extend(list(np.setdifff1d(np.arange(filtered_df.shape[0]), list(df_.index))))
                    else:
                        matchings_idxs = matching_idxs.extend(list(df_.index))

                filtered_df = filtered_df.iloc[matching_idxs]
        
            elif type(value) == str:
                filtered_df = filtered_df.loc[[value in s for s in filtered_df[key].values]]
            else:
                if invert:
                    filtered_df = filtered_df.loc[filtered_df[key] != value]
                else:
                    filtered_df = filtered_df.loc[filtered_df[key] == value]
        
        if reset_index:
            filtered_df.reset_index(inplace=True, drop=True)

        # if filtered_df.shape[0] == 0:
        #     print('Key %s reduced size to 0!' % key)

    return filtered_df

def get_data_path(region):
    root_path = PATH_DICT['data']
    if region in ['M1', 'S1', 'M1_trialized', 'M1_psid', 'S1_psid']:
        # root_path = '/home/ankit_kumar/Data'
        data_path = 'sabes'
    elif region == 'HPC_peanut':
        data_path = 'peanut/data_dict_peanut_day14.obj'
    elif region in ['VISp']:
        data_path = 'AllenData'
    return root_path + '/' + data_path

def load_supervised_decoding_df(region, **kwargs):
    root_path = PATH_DICT['df']
    if region == 'M1_psid':
        if kwargs['use_highd']:
            with open(root_path + '/sabes_highd_sup_decodingdf.pkl', 'rb') as f:
                rl = pickle.load(f)
        else:
            with open(root_path + '/sabes_supervised_df.pkl', 'rb') as f:
                rl = pickle.load(f)
        df = pd.DataFrame(rl)
        dims = df.iloc[0]['decoder_args']['state_dim']
    elif region == 'S1_psid':
        if kwargs['use_highd']:
            with open(root_path + '/sabes_highd_sup_decodingdfS1.pkl', 'rb') as f:
                rl = pickle.load(f)
        else:
            with open(root_path + '/sabes_supervised_dfS1.pkl', 'rb') as f:
                rl = pickle.load(f)
        df = pd.DataFrame(rl)
        dims = df.iloc[0]['decoder_args']['state_dim']
    elif region == 'HPC_peanut':
        with open(root_path + '/peanut_supervised_decoding25_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)    
        # Add epoch as a top level key and remove from loader args
        epochs = [df.iloc[k]['loader_args']['epoch'] for k in range(df.shape[0])]
        df['epoch'] = epochs
        # Pop epoch as a key in loader args to prevent double passing this 
        # argument down the line
        for k in range(df.shape[0]):
            try:
                del df.iloc[k]['loader_args']['epoch']
            except KeyError:
                pass

        filt = [idx for idx in range(df.shape[0])
                if df.iloc[idx]['decoder_args']['decoding_window'] == 12]
        df = df.iloc[filt]
        dims = None
    elif region == 'VISp':
        with open(root_path + '/visp_logreg_svd.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        dims = np.unique(df['dim'].values)
    return df, dims


def load_rand_decoding_df(region, **kwargs):
    root_path = PATH_DICT['df']
    if region == 'M1_psid':
        with open(root_path + '/sabes_rand_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df2 = pd.DataFrame(rl)
        df2 = apply_df_filters(df2, dim=[2, 4])

        # # Dimensions 30+
        # with open(root_path + '/sabes_highd_rand_decoding_df.pkl', 'rb') as f:
        #     rl = pickle.load(f)
        # df_highd = pd.DataFrame(rl)

        with open(root_path + '/sabes_rand_decodingdf_v2.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)

        df = pd.concat([df, df2])

    elif region == 'S1_psid':

        # Get d = 2, 4
        with open(root_path + '/sabes_rand_decoding_dfS1.pkl', 'rb') as f:
            rl = pickle.load(f)
        df2 = pd.DataFrame(rl)
        df2 = apply_df_filters(df2, dim=[2, 4])

        # # Dimensions 30+
        # with open(root_path + '/sabes_highd_rand_decoding_dfS1.pkl', 'rb') as f:
        #     rl = pickle.load(f)
        # df_highd = pd.DataFrame(rl)
        # df = pd.concat([df, df_highd])

        # Higher dimensions
        with open(root_path + '/sabes_rand_decodingdf_v2S1.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        df = pd.concat([df, df2])

    elif region == 'HPC_peanut':
        with open(root_path + '/peanut_rand_decoding25_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)    
        # Add epoch as a top level key and remove from loader args
        epochs = [df.iloc[k]['loader_args']['epoch'] for k in range(df.shape[0])]
        df['epoch'] = epochs
        # Pop epoch as a key in loader args to prevent double passing this 
        # argument down the line
        for k in range(df.shape[0]):
            try:
                del df.iloc[k]['loader_args']['epoch']
            except KeyError:
                pass
        filt = [idx for idx in range(df.shape[0])
                if df.iloc[idx]['decoder_args']['decoding_window'] == 12]
        df = df.iloc[filt]
    elif region in ['VISp']:
        with open(root_path + '/allen_rand_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
    return df

def load_decoding_df(region, **kwargs):
    root_path = PATH_DICT['df']
    if region == 'M1':
        with open(root_path + '/sabes_m1subtrunc_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)

        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        df = df.iloc[filt]
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        df = pd.concat([df_pca, df_fcca])
        # filter by start time truncation and subset selection
        # filt = [idx for idx in range(df.shape[0]) 
        #         if df.iloc[idx]['loader_args']['subset'] is not None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        # df = df.iloc[filt]
        session_key = 'data_file'
    elif region == 'M1_psid':
        if kwargs['use_highd']:
            with open(root_path + '/sabes_highd_decoding_df.pkl', 'rb') as f:
                rl = pickle.load(f)
            df = pd.DataFrame(rl)
        else:
            with open(root_path + '/sabes_psid_decoding_dfM1.pkl', 'rb') as f:
                rl = pickle.load(f)

            df = pd.DataFrame(rl)

            # Filter by start time truncation only
            filt = [idx for idx in range(df.shape[0]) 
                    if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
            df = df.iloc[filt]
            df_pca = apply_df_filters(df, dimreduc_method='PCA')
            df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
            df = pd.concat([df_pca, df_fcca])

        session_key = 'data_file'

    elif region in ['S1']:
        
        with open(root_path + '/sabes_s1subtrunc_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)

        df = pd.DataFrame(rl)

        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        df = df.iloc[filt]


        # filter by start time truncation and subset selection
        # filt = [idx for idx in range(df.shape[0]) 
        #         if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        # df = df.iloc[filt]

        # Filter by decoder args
        filt = [idx for idx in range(df.shape[0])
                if df.iloc[idx]['decoder_args']['trainlag'] == 2]
        df = df.iloc[filt]
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        df = pd.concat([df_pca, df_fcca])
        session_key = 'data_file'

    elif region == 'S1_psid':

        if kwargs['use_highd']:
            with open(root_path + '/sabes_highd_decoding_dfS1.pkl', 'rb') as f:
                rl = pickle.load(f)
            df = pd.DataFrame(rl)
        else:
            with open(root_path + '/sabes_psid_decoding_dfS1.pkl', 'rb') as f:
                rl = pickle.load(f)
            df = pd.DataFrame(rl)
            filt = [idx for idx in range(df.shape[0]) 
                    if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
            df = df.iloc[filt]
            df_pca = apply_df_filters(df, dimreduc_method='PCA')
            df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
            df = pd.concat([df_pca, df_fcca])
        session_key = 'data_file'

    elif region == 'HPC_peanut':
        with open(root_path + '/peanut_decoding25_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        # Need to add epoch as a top level key
        epochs = [df.iloc[k]['loader_args']['epoch'] for k in range(df.shape[0])]
        df['epoch'] = epochs

        # Filter arguments
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':5, 'n_init':10, 'rng_or_seed':42})
        df = pd.concat([df_pca, df_fcca])
        filt = [idx for idx in range(df.shape[0])
                if df.iloc[idx]['decoder_args']['decoding_window'] == 12]
        df = df.iloc[filt]
        
        # Pop epoch as a key in loader args to prevent double passing this 
        # argument down the line
        for k in range(df.shape[0]):
            try:
                del df.iloc[k]['loader_args']['epoch']
            except KeyError:
                pass
        session_key = 'epoch'


    elif region == 'VISp':
        # Get entire DF
        with open(root_path + '/decoding_AllenVC_VISp_glom.pickle', 'rb') as f:            
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        session_key = 'data_file'
                
        # Filter it for certain load params (if applicable)
        if 'load_idx' not in kwargs:
            load_idx = loader_kwargs[region]['load_idx']
        else:
            load_idx = kwargs['load_idx']
            
        unique_loader_args = list({frozenset(d.items()) for d in df['loader_args']})
        df = apply_df_filters(df, loader_args=dict(unique_loader_args[load_idx]))
                
        
    return df, session_key

def load_data(data_path, region, session, loader_args, full_arg_tuple=None):
    if region in ['M1', 'M1_trialized', 'S1', 'M1_psid', 'S1_psid']:
        data_file = session
        if 'region' not in loader_args:
            loader_args['region'] = region
        loader_args['high_pass'] = True
        dat = load_sabes('%s/%s' % (data_path, data_file), **loader_args)            
    elif region == 'HPC_peanut':
        epoch = session
        dat = load_peanut(data_path, epoch=epoch, **loader_args)
    elif region in ['VISp']:        
        sess_folder = session.split(".")[0]
        path_to_data = data_path + '/' + sess_folder + "/" + session
                
        dat = load_AllenVC(path_to_data, **loader_args)
    return dat

def get_rates_smoothed(data_path, region, session, trial_average=True,
                       std=False, boxcox=False, full_arg_tuple=None, 
                       loader_args=None, return_t=False, sigma=2):
    if boxcox:
        boxcox = 0.5
    else:
        boxcox = None

    if region in ['M1', 'S1', 'M1_trialized']:
        data_file = session
        if region == 'M1_trialized':
            dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=boxcox, high_pass=False, region='M1')
        else:
            dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=boxcox, high_pass=False, region=region)
        dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 50 * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = dat['spike_rates'].shape[-1]

        if trial_average:
            x = np.zeros((n, time.size))
        else:
            x = np.zeros((n,), dtype=object)
        for j in range(n):
            x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, j] 
                        for idx in valid_transitions])
            if std:
                x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=sigma)
            if trial_average:
                x_ = np.mean(x_, axis=0)
            x[j] = x_               

    elif region == 'HPC_peanut':
        epoch = session 
        dat = load_peanut(data_path, epoch=epoch, boxcox=boxcox, spike_threshold=100, bin_width=25)
        loc_file_path = '/'.join(data_path.split('/')[:-1])
        transitions = segment_peanut(dat, loc_file=loc_file_path + '/linearization_dict_peanut_day14.obj', 
                                     epoch=epoch)

        # For now, aggregate all types of transitions together.
        transitions_all = transitions[0]
        transitions_all.extend(transitions[1])

        lens = [len(t) for t in transitions_all]
        T = 100
        n = dat['spike_rates'].shape[-1]
        time = 25 * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = dat['spike_rates'].shape[-1]
        if trial_average:
            x = np.zeros((n, time.size))
        else:
            x = np.zeros((n,), dtype=object)
        for j in range(n):
            # Are all trials longer than T?
            assert(np.all([len(t) > T for t in transitions_all]))
            x_ = np.array([dat['spike_rates'][trans[0]:trans[0] + T, j] 
                        for trans in transitions_all])

            if std:
                x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=sigma)
            if trial_average:
                x_ = np.mean(x_, axis=0)
            x[j] = x_

    elif region in ['VISp']:
        
        sess_folder = session.split(".")[0]
        path_to_data = data_path + '/' + sess_folder + "/" + session
        loader_args['boxcox'] = boxcox
        dat = load_AllenVC(path_to_data, **loader_args)
        
        x = dat['spike_rates']
        x = gaussian_filter1d(x, sigma=sigma, axis=1)

        # if zscore: spike_rates = zcore_spikes(dat, region)
        if std:
            y = StandardScaler().fit_transform(x.reshape(-1, x.shape[-1]))
            x = y.reshape(x.shape)
        
        if trial_average:
            x = np.mean(x, axis=0).squeeze().T # Averaged spike rates, shape: (numUnits, time)
        else:
            x = x.transpose((2, 0, 1)) # reshape to n_neurons, n_trials, n_time
        
        
        T = x.shape[-1]
        time = loader_args['bin_width'] * np.arange(T)
                
    if return_t:
        return x, time
    else:   
        return x

    
def get_rates_raw(data_path, region, session, loader_args=None, full_arg_tuple=None, zscore=False):
    
    dat = load_data(data_path, region, session, loader_args, full_arg_tuple)
    
    if region in ['M1', 'S1']:
        dat_segment = reach_segment_sabes(dat, data_file=session.split('.mat')[0])
        spike_rates = [dat['spike_rates'].squeeze()[t0:t1] 
                        for t0, t1 in dat_segment['transition_times']]
    elif region in ['HPC_peanut']:
        loc_file_path = '/'.join(data_path.split('/')[:-1])
        transitions = segment_peanut(dat, loc_file=loc_file_path + '/linearization_dict_peanut_day14.obj', 
                                     epoch=session)

        # For now, aggregate all types of transitions together.
        transitions_all = transitions[0]
        transitions_all.extend(transitions[1])
        spike_rates = [dat['spike_rates'][trans[0]:trans[-1], :] for trans in transitions_all]

    elif region in ['VISp']:
        spike_rates = dat['spike_rates']
        
    return spike_rates

def make_hashable(d):
    """Recursively convert unhashable elements (dicts/lists) into hashable types."""
    if isinstance(d, dict):
        return frozenset((k, make_hashable(v)) for k, v in d.items())  # Convert dict to frozenset
    elif isinstance(d, list):
        return tuple(make_hashable(v) for v in d)  # Convert list to tuple
    elif isinstance(d, np.ndarray):  
        return tuple(d.tolist())  # Convert NumPy array to tuple
    else:
        return d  # Keep other types unchanged
