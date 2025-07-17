import glob
import os
import numpy as np
import pdb


script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/peanut'

desc = 'Peanut dimreduc at 25 ms bins'
data_files = ['%s/data_dict_peanut_day14.obj' % data_path]

loader = 'peanut'
analysis_type = 'dimreduc'

loader_args = [{'bin_width':25, 'epoch': epoch, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'speed_threshold':4, 'region':'HPC'}
               for epoch in np.arange(2, 6, 2)]


dimvals = np.arange(1, 3)
task_args = [{'dim_vals':dimvals, 'n_folds':2, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'n_init':2, 'rng_or_seed':0}},
              {'dim_vals':dimvals, 'n_folds':2, 'dimreduc_method':'PCA', 'dimreduc_args':{}}]
