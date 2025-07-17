import numpy as np

script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/peanut'
data_files = ['%s/data_dict_peanut_day14.obj' % data_path]

loader = 'peanut'
analysis_type = 'dimreduc'
desc = 'Peanut dimreduc at 25 ms bins'


loader_args = [{'bin_width':25, 'epoch': epoch, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 
                'spike_threshold':100, 'speed_threshold':4, 'region':'HPC'}
               for epoch in np.arange(2, 18, 2)]

dimvals = np.arange(1, 31)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 
              'dimreduc_args': {'T':5, 'n_init':10, 'rng_or_seed':42}},
              {'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'PCA',
              'dimreduc_args':{}}]
