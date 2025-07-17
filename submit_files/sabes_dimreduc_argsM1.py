import glob
import numpy as np


script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/sabes' 


loader = 'sabes'
analysis_type = 'dimreduc'
desc = 'Dimreduc on Sabes M1 data' 


data_files = glob.glob('%s/indy*.mat' % data_path)
good_loco_files = ['loco_20170210_03.mat',
            'loco_20170213_02.mat',
            'loco_20170215_02.mat',
            'loco_20170227_04.mat',
            'loco_20170228_02.mat',
            'loco_20170301_05.mat',
            'loco_20170302_02.mat']

for glf in good_loco_files:
    data_files.append('%s/%s' % (data_path, glf))

loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1',
                'truncate_start':True, 'subset':None}]
dimvals = np.arange(1, 81)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {}},
             {'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 
              'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42}}]