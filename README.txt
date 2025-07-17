The repository contains an environment.yml file, which can be used with conda/mamba/micromamba to initialize a virtual enviornment.

Scripts to re-create main figures from the manuscript are contained in analysis_scripts. Some environment configuration is required. Edit analysis_scripts/config.py to point to the correct paths.
In particular, figure re-creation requires access to dataframes containing fits of FCCA/PCA to neural data as well as associated fits of decoders to the projected neural data. These dataframes
may be accesssed from here: https://www.dropbox.com/scl/fo/avk0dh74iul61mla104nb/AFb-iGE5OERu-p1E_7FBY24?rlkey=oha56po1rk9q3toovsc89wnm7&st=a8wosj65&dl=0

The plots in Fig5.py depend on the outputs of calc_su_statistics.py. Similarly, the plots in Fig6.py depend on the outputs of fig6calcs.py. Subfolder fcca_analysis_tmp 
in the link above contains intermediate calculations and may be used by setting the tmp path in config.py

To rerun fits of FCCA on neural data:

1. Make a filname_dimreduc_args.py file in submit_files
    - set paths to data and this repo   
    - set parameters for dimensionality reeduction
    - save 
2. Go to RunDimreduc.py 
    - put in path to the dimreduc_args.py file (can be found in submit_files folder for each dataset)
    - set output folder (this is where the code will be ran and outputs stored)
    - Run the script 
3. (Opt.) Go to the folder where outputs are stored
    - Check log.txt to see run progress
    - If there are any issues, can make edits to the code, and continue the run with "nohup ./sbatch_resume.sh >> log.txt 2>&1 &"
    which will launch the scripts again in the background.
4. After dimreduc step has completed, use RunDecoding.py to run the decoding step. The corresponding *_decoding_args.py submit file will require the path
    to the output of the dimreduc step prior to being ingested by RunDecoding.py
5. After the decoding step has completed, use ConsolidateStructs.py to put all dataframes into one object, which are used
    for downstream analyses. You may point the df_root_path variable in analysis_scripts/config.py to the location of the refit dataframes.
