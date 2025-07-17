import glob
import numpy as np
import itertools
import os

script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
desc = 'Decoding using PSID on sabes high d'
data_path = '/clusterfs/NSDS_data/FCCA/data/sabes'    

data_files = [' ']
loader = 'sabes'
analysis_type = 'decoding'
loader_args = [[]]
dimreduc_folder = '/clusterfs/NSDS_data/FCCA/testDFs/LocoNew_dr'
dimreduc_files = glob.glob(os.path.join(dimreduc_folder, f'{os.path.basename(dimreduc_folder)}_*.dat'))

decoders = [{'method': 'psid', 'args':{'lag': 5}}]
task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})