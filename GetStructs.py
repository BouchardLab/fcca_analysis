import time
import os
import glob
from batch_util import init_batch, launch_batch


dim_jobdir = '/clusterfs/NSDS_data/FCCA/testDFs/AllenNew_dr'
dec_jobdir = '/clusterfs/NSDS_data/FCCA/testDFs/AllenNew_dc'

dim_submit_file = '/home/marcush/projects/neural_control/submit_files/test_submit_files/AllenVC_dimreduc_args.py'
dec_submit_file = '/home/marcush/projects/neural_control/submit_files/test_submit_files/AllenVC_decoding_args.py'

RunSerial = False


# --------------------------
# 1. Dimreduc Phase
# --------------------------
print("Running Dimreduc...")

init_batch(dim_submit_file, dim_jobdir, local=True, serial=RunSerial)
launch_batch(dim_jobdir)

RUN CONSOLIDATION

# --------------------------
# 2. Wait until all dimreduc results exist
# --------------------------
print("Waiting for all dimreduc results to finish...")

while True:
    arg_files = sorted([f for f in os.listdir(dim_jobdir) if f.startswith("arg") and f.endswith(".dat")])
    job_nums = [int(f[3:-4]) for f in arg_files]  # extract N from argN.dat
    base_name = os.path.basename(dim_jobdir)  # e.g., "AllenNew_dc"

    results_done = all(
        os.path.exists(os.path.join(dim_jobdir, f"{base_name}_{n}.dat")) and
        os.path.isdir(os.path.join(dim_jobdir, f"{base_name}_{n}")) and
        len(glob.glob(os.path.join(dim_jobdir, f"{base_name}_{n}", "*.dat"))) > 0
        for n in job_nums
    )
    if results_done:
        break
    time.sleep(60)  # check every minute

# --------------------------
# 3. Decoding Phase
# --------------------------
print("Running Decoding...")

init_batch(dec_submit_file, dec_jobdir, local=True, serial=RunSerial)
launch_batch(dec_jobdir)