import os
from batch_util import init_batch


#jobdir = '/clusterfs/NSDS_data/FCCA/testDFs/LocoNew_dr'
#submit_file = '/home/marcush/projects/neural_control/submit_files/test_submit_files/loco_dimreduc_args.py'
jobdir = '/clusterfs/NSDS_data/FCCA/testDFs/AllenNew_dc'
submit_file = '/home/marcush/projects/neural_control/submit_files/test_submit_files/AllenVC_decoding_args.py'


# Initializes the paths/folders for analysis
init_batch(submit_file, jobdir, local=True)
#init_batch(submit_file, jobdir, local=True, sequential=True, serial=True)

