import time
import os
import glob
from batch_util import init_batch, launch_batch


dec_jobdir = '/clusterfs/NSDS_data/FCCA/testDFs/AllenNew_dc'
dec_submit_file = '/home/marcush/projects/neural_control/submit_files/AllenVC_decoding_args.py'
RunSerial = False


init_batch(dec_submit_file, dec_jobdir, local=True, serial=RunSerial)
launch_batch(dec_jobdir)
