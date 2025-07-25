#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
import pdb
import scipy
import pickle
from config import PATH_DICT
from region_select import *
from dca.methods_comparison import JPCA

def get_rates_largs(T, df, data_path, region, session):
    if region == 'HPC_peanut':
        loader_args = {'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}
    else:
        loader_args = df.iloc[0]['loader_args']

    if region in ['ML', 'AM']:
        df_ = apply_df_filters(df, **{'data_file':session, 'loader_args':{'region': region}})
        y = get_rates_raw(data_path, region, session, loader_args, full_arg_tuple=df_['full_arg_tuple'])
    
    elif region in ['VISp']:

        #df_ = apply_df_filters(df, **{'data_file':session, 'loader_args':{'region': region, 'bin_width': 15, 'preTrialWindowMS': 0, 'postTrialWindowMS': 0}})
        #loader_args = df_.iloc[0]['loader_args']
        load_idx = 0
        unique_loader_args = list({frozenset(d.items()) for d in df['loader_args']})
        loader_args=dict(unique_loader_args[load_idx])
        #df = apply_df_filters(df, loader_args=dict(unique_loader_args[load_idx]))
        y = get_rates_raw(data_path, region, session, loader_args)
    else:
        y = get_rates_raw(data_path, region, session, loader_args)

    # Restrict to trials that match the length threshold, and standardize lengths
    y = np.array([y_[0:T] for y_ in y if len(y_) > T])
    return y


def calc_on_dimreduc(T, decoding_df, region, session_key, DIM):
    jDIM = DIM - 1 if DIM % 2 != 0 else DIM    # jPCA dimension must be even
    
    sessions = np.unique(decoding_df[session_key].values)
    data_path = get_data_path(region)
    results = []
    for ii, session in enumerate(sessions):
        y = get_rates_largs(T, decoding_df, data_path, region, session)
        for dimreduc_method in [['LQGCA', 'FCCA'], 'PCA']:
            if region in ['AM', 'ML']:
                df_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 
                            'dimreduc_method':dimreduc_method, 'loader_args':{'region':region}}
                df_ = apply_df_filters(decoding_df , **df_filter)
            else:
                df_filter = {session_key:session, 'fold_idx':0, 'dim':DIM,
                             'dimreduc_method':dimreduc_method}
                df_ = apply_df_filters(decoding_df, **df_filter)
            assert(df_.shape[0] == 1)

            V = df_.iloc[0]['coef']
            if dimreduc_method == 'PCA':
                V = V[:, 0:jDIM]        

            yproj = y @ V
            result_ = {}
            result_[session_key] = session
            result_['dimreduc_method'] = dimreduc_method

            # 3 fits: Look at symmetric vs. asymmetric portions of regression onto differences
            jpca = JPCA(n_components=jDIM, mean_subtract=False)
            jpca.fit(yproj)
            
            result_['jeig'] = jpca.eigen_vals_
            yprojcent = yproj
            dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
            result_['dyn_range'] = np.mean(dyn_range)

            # Now also try on a normalized version of the data
            # We want the normalization to be common within dataset so that different datasets
            # are put on the same scale. Therefore, let' try two normalization stratgies -
            # (1) Normalize by the max value of y across neurons.
            # (2) Normalize by the standard deviation of y across all neurons

            ynorm1 = y / np.max(y)
            ynorm2 = y / np.std(y)

            yproj1 = ynorm1 @ V
            yproj2 = ynorm2 @ V

            jpca1 = JPCA(n_components=jDIM, mean_subtract=False)
            jpca1.fit(yproj1)
            jpca2 = JPCA(n_components=jDIM, mean_subtract=False)
            jpca2.fit(yproj2)

            result_['jeig_maxnorm'] = jpca1.eigen_vals_
            result_['jeig_stdnorm'] = jpca2.eigen_vals_

            yprojcent = yproj1
            dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
            result_['dyn_range_maxnorm'] = np.mean(dyn_range)

            yprojcent = yproj2
            dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
            result_['dyn_range_stdnorm'] = np.mean(dyn_range)

            results.append(result_)

    save_path = PATH_DICT['tmp'] + f'/jpca_tmp_dimreduc_{region}_dim{DIM}_T{T}.pkl'    
    with open(save_path, 'wb') as f:
        f.write(pickle.dumps(results))                

def calc_on_random(T, decoding_df, region, session_key, DIM, inner_reps):
    jDIM = DIM - 1 if DIM % 2 != 0 else DIM    # jPCA dimension must be even

    results = []
    sessions = np.unique(decoding_df[session_key].values)
    data_path = get_data_path(region)
    for ii, session in enumerate(sessions):
        y = get_rates_largs(T, decoding_df, data_path, region, session)
        # Randomly project the spike rates and fit JPCA
        for j in tqdm(range(inner_reps)):
            V = scipy.stats.special_ortho_group.rvs(y.shape[-1], random_state=np.random.RandomState(j))
            V = V[:, 0:jDIM]
            # Project data
            yproj = y @ V
            result_ = {}
            result_[session_key] = session
            result_['inner_rep'] = j

            jpca = JPCA(n_components=jDIM, mean_subtract=False)
            jpca.fit(yproj)
            result_['jeig'] = jpca.eigen_vals_

            yprojcent = np.array([y_ - y_[0:1, :] for y_ in yproj])
            dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
            result_['dyn_range'] = np.mean(dyn_range)

            ynorm1 = y / np.max(y)
            ynorm2 = y / np.std(y)

            yproj1 = ynorm1 @ V
            yproj2 = ynorm2 @ V

            jpca1 = JPCA(n_components=jDIM, mean_subtract=False)
            jpca1.fit(yproj1)
            jpca2 = JPCA(n_components=jDIM, mean_subtract=False)
            jpca2.fit(yproj2)

            result_['jeig_maxnorm'] = jpca1.eigen_vals_
            result_['jeig_stdnorm'] = jpca2.eigen_vals_

            yprojcent = yproj1
            dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
            result_['dyn_range_maxnorm'] = np.mean(dyn_range)

            yprojcent = yproj2
            dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
            result_['dyn_range_stdnorm'] = np.mean(dyn_range)

            results.append(result_)
            
            print(f"Done with inner rep {j}")

    save_path = PATH_DICT['tmp'] + f'/jpca_tmp_randcontrol_{region}_dim{DIM}_T{T}.pkl'    
    with open(save_path, 'wb') as f:
        f.write(pickle.dumps(results))                

T_dict = {
    'M1': 20,
    'S1': 20,
    'HPC_peanut': 40,
    'VISp': 14,
}

dim_dict =  {
        'M1': 6,
        'S1': 6,
        'HPC_peanut': 6,
        'VISp': 6
    }

if __name__ == '__main__':

    regions = ['M1', 'S1', 'HPC_peanut', 'VISp']

    print("Beginning..")
    for region in regions:
        T = T_dict[region]
        inner_reps = 1000
        decoding_df, session_key = load_decoding_df(region, **loader_kwargs[region])
        calc_on_dimreduc(T, decoding_df, region, session_key, dim_dict[region])
        calc_on_random(T, decoding_df, region, session_key, dim_dict[region], inner_reps)
        print(f"Done with {region}")