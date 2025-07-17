import glob
import os
import numpy as np

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECK MARGINALS FLAG BELOW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/AllenData'


loader = 'AllenVC'
analysis_type = 'dimreduc'
Region = 'VISp'  
desc = 'test desc'


session_IDs = [732592105, 754312389]
data_files = [os.path.join(data_path, f"session_{session_ID}", f"session_{session_ID}.nwb") for session_ID in session_IDs]


loader_args = [{'region': Region, 'bin_width':25, 'preTrialWindowMS':50, 'postTrialWindowMS':100, 'boxcox':0.5},
               {'region': Region, 'bin_width':15, 'preTrialWindowMS':0, 'postTrialWindowMS':0, 'boxcox':0.5}]


KFold = 2
dimvals = np.arange(1,3)
task_args = [{'dim_vals':dimvals, 'n_folds':KFold, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':2, 'rng_or_seed':0}},
             {'dim_vals':dimvals, 'n_folds':KFold, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]

