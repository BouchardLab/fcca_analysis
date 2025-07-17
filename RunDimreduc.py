from batch_util import init_batch, launch_batch


dim_jobdir = '/clusterfs/NSDS_data/FCCA/finalDFs/peanut_dr'
dim_submit_file = '/home/marcush/projects/fcca_analysis/submit_files/peanut_dimreduc_args.py'
RunSerial = False


init_batch(dim_submit_file, dim_jobdir, local=True, serial=RunSerial)
out = launch_batch(dim_jobdir)
print(out)  