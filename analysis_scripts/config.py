# What is the root path of the fcca_analysis repo?
repo_root_path = '/home/ankit_kumar/fcca_analysis'

# Where is the neural data stored?
data_root_path = '/clusterfs/NSDS_data/FCCA/data'

# Where are the decoding datarames stored?
df_root_path = '/clusterfs/NSDS_data/FCCA/postprocessed'

# Where should tmp outputs of analysis scripts be stored/looked for?
tmp_root_path = '/clusterfs/NSDS_data/FCCA/tmp_reproduction'

# Where should figures be saved to?
fig_path = '/clusterfs/NSDS_data/FCCA/figs_reproduction'

# Where are SOC results located?
soc_path = '/clusterfs/NSDS_data/FCCA/soc'

# Where T.O. RNN results located?
rnn_path = '/clustefs/NSDS_data/EMG_analysis'

PATH_DICT = {
    'data': data_root_path,
    'df': df_root_path,
    'tmp': tmp_root_path,
    'figs': fig_path,
    'repo': repo_root_path,
    'soc': soc_path,
    'rnn': rnn_path
}