import time
import os
import glob
from batch_util import init_batch, launch_batch


dim_jobdir = '/clusterfs/NSDS_data/FCCA/testDFs/AllenNew_dr'
dim_submit_file = '/home/marcush/projects/neural_control/submit_files/AllenVC_dimreduc_args.py'
RunSerial = False


init_batch(dim_submit_file, dim_jobdir, local=True, serial=RunSerial)
launch_batch(dim_jobdir)
