from consolidation import consolidate_decoding, consolidate_dimreduc


df = "sabes_S1"

dimreduc_file_path = f'/clusterfs/NSDS_data/FCCA/finalDFs/{df}_dr'
dimreduc_output_path = f'/clusterfs/NSDS_data/FCCA/finalDFs/{df}_dr/{df}_dr_final.pickle'

consolidate_dimreduc(dimreduc_file_path, dimreduc_output_path)


decode_file_path = f'/clusterfs/NSDS_data/FCCA/finalDFs/{df}_dc'
decode_output_path = f'/clusterfs/NSDS_data/FCCA/finalDFs/{df}_dc/{df}_dc_final.pickle'

consolidate_decoding(decode_file_path, decode_output_path)
 