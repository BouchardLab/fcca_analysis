import glob
import os
import itertools


script_path = '/home/marcush/projects/fcca_analysis/batch_analysis.py'
data_path = '/clusterfs/NSDS_data/FCCA/data/peanut'
dimreduc_folder = '/clusterfs/NSDS_data/FCCA/testDFs/PeanutNew_dr'
dimreduc_files = glob.glob(os.path.join(dimreduc_folder, f'{os.path.basename(dimreduc_folder)}_*.dat'))

data_files = ['']
desc = 'Decoding from 25 ms'

loader = 'peanut'
analysis_type = 'decoding'
loader_args = [[]]


decoders = [{'method': 'lr', 'args':{'trainlag': 0, 'testlag': 0, 'decoding_window': 6}}]

task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})
