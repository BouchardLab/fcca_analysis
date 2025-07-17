from batch_util import init_batch, launch_batch


dec_jobdir = '/clusterfs/NSDS_data/FCCA/finalDFs/peanut_dc'
dec_submit_file = '/home/marcush/projects/fcca_analysis/submit_files/peanut_decoding_args.py'
RunSerial = False


init_batch(dec_submit_file, dec_jobdir, local=True, serial=RunSerial)
out = launch_batch(dec_jobdir)
print(out) 