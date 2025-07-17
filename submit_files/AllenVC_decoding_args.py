import glob
import itertools
import os

 
data_path = '/clusterfs/NSDS_data/FCCA/data/AllenData' 
script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'

dimreduc_folder = '/clusterfs/NSDS_data/FCCA/finalDFs/AllenVC_dr'
dimreduc_files = glob.glob(os.path.join(dimreduc_folder, f'{os.path.basename(dimreduc_folder)}_*.dat'))


loader = 'AllenVC'
analysis_type = 'decoding'
desc = "VISp Decoding"


data_files = [' ']
loader_args = [[]]


dimreduc_files = [file for file in dimreduc_files if not os.path.basename(file).startswith('arg')]
decoders = [{'method': 'logreg', 'args':{}}]

task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})
