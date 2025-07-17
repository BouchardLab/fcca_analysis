import os
import numpy as np

""""
Use Description ::

 - Add " 'marginal_only':True " to dimreduc_args for PCA and LQGCA in order to get marginals. 

"""

# ------------------------------------------------------------------ Set Paths
script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/AllenData' 

loader = 'AllenVC'
Region = 'VISp'  

desc = 'Allen Visual Coding Dimreduc on sessions with >50 units for area VISp'
analysis_type = 'dimreduc'

# ------------------------------------------------------------------ Set Parameters
KFold = 5
dimvals = np.arange(1,48)
num_time_points = 3
n_init = 5
rng_seed = 0

session_IDs = [732592105, 754312389, 798911424, 791319847, 754829445, 760693773, 757216464, 797828357, 762120172, 757970808, 799864342, 762602078, 755434585, 763673393, 760345702, 750332458, 715093703, 759883607, 719161530, 750749662, 756029989]
data_files = [os.path.join(data_path, f"session_{session_ID}", f"session_{session_ID}.nwb") for session_ID in session_IDs]


# Each of these can be made into a list whose outer product is taken
loader_args = [{'region': Region, 'bin_width':25, 'preTrialWindowMS':50, 'postTrialWindowMS':100, 'boxcox':0.5},
               {'region': Region, 'bin_width':15, 'preTrialWindowMS':0, 'postTrialWindowMS':0, 'boxcox':0.5}]


task_args = [{'dim_vals':dimvals, 'n_folds':KFold, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':num_time_points, 'loss_type':'trace', 'n_init':n_init, 'rng_or_seed':rng_seed}},
             {'dim_vals':dimvals, 'n_folds':KFold, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]







