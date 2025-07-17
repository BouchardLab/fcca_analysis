import os
import glob
import itertools


script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/sabes' 

dimreduc_folder = '/clusterfs/NSDS_data/FCCA/finalDFs/sabes_M1_dr'
dimreduc_files = glob.glob(os.path.join(dimreduc_folder, f'{os.path.basename(dimreduc_folder)}_*.dat'))


loader = 'sabes'
analysis_type = 'decoding'
decoders = [{'method': 'psid', 'args':{'lag': 5}}]
desc = 'Decoding from M1 high d using PSID'


data_files = [' ']
loader_args = [[]]


task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})
