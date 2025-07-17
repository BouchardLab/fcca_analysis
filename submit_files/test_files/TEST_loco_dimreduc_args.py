import glob
import numpy as np

script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/sabes'    

desc = 'Loco dimreduc with corrected version of FCCA, single fold'
 
data_files = glob.glob('%s/loco_201703*' % data_path)

loader = 'sabes'
analysis_type = 'dimreduc'

 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'S1'}]

n_folds=2
task_args = [{'dim_vals':np.arange(1, 3), 'n_folds':n_folds, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':2}},
             {'dim_vals':np.arange(1, 3), 'n_folds': n_folds, 'dimreduc_method':'PCA', 'dimreduc_args':{}}]