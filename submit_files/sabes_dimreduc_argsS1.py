import numpy as np


script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/sabes' 
desc = 'Dimreduc on Sabes S1 data' 
 
fls = ['loco_20170210_03.mat',
 'loco_20170213_02.mat',
 'loco_20170215_02.mat',
 'loco_20170227_04.mat',
 'loco_20170228_02.mat',
 'loco_20170301_05.mat',
 'loco_20170302_02.mat',
 'indy_20160426_01.mat']

data_files = []
for f in fls:
    data_files.append('%s/%s' % (data_path, f))

loader = 'sabes'
analysis_type = 'dimreduc'

loader_args = []
loader_args.append({'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'S1',
                    'truncate_start':True})
dimvals = np.arange(1, 81)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {}},
             {'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 
              'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42}}]