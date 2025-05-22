import os 
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pickle
import pandas as pd
import warnings
import pdb
from tqdm import tqdm

region="VISp"
bin_width=25
preTrialWindowMS=50
postTrialWindowMS=100

DataFolderPath = '/clusterfs/NSDS_data/FCCA/data/AllenData'
output_path = '/clusterfs/NSDS_data/FCCA/data/AllenData/preloaded_spikes'
# Get unique sessions
# root_path = '/clusterfs/NSDS_data/FCCA/postprocessed'
# with open(root_path + '/decoding_AllenVC_VISp_glom.pickle', 'rb') as f:            
#     rl = pickle.load(f)
# df = pd.DataFrame(rl)
# session_key = 'data_file'

# Unique sessions hard coded from the data frame
sessions = np.array(['session_715093703.nwb', 'session_719161530.nwb',
       'session_732592105.nwb', 'session_750332458.nwb',
       'session_750749662.nwb', 'session_754312389.nwb',
       'session_754829445.nwb', 'session_755434585.nwb',
       'session_756029989.nwb', 'session_757216464.nwb',
       'session_757970808.nwb', 'session_759883607.nwb',
       'session_760345702.nwb', 'session_760693773.nwb',
       'session_762120172.nwb', 'session_762602078.nwb',
       'session_763673393.nwb', 'session_791319847.nwb',
       'session_797828357.nwb', 'session_798911424.nwb',
       'session_799864342.nwb'])

manifest_path = os.path.join(DataFolderPath, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
for session in tqdm(sessions):
    session_id = int(session.split('_')[1].split('.')[0])
    # For all session info, including regions, stimulus names, unit count, etc. see: session.metadata
    warnings.filterwarnings("ignore", category=UserWarning)
    session = cache.get_session_data(session_id)
        
    units = session.units[session.units["ecephys_structure_acronym"] == region]
    # if units.empty: return {} # Check that this region is in this session and has units

    presentations = session.get_stimulus_table("natural_scenes") 
    stimIDs = presentations.loc[:, "frame"].values.astype(int) # Per trial stimulus IDs

    # Pre-, and post- trial windows are in units of ms. Convert to seconds
    binarize_bin = 1/1000 # 1ms bins in units of seconds
    DefaultTrialDuration = 0.25 # units of seconds
    time_bins = np.arange(-(preTrialWindowMS/1000), DefaultTrialDuration + (postTrialWindowMS/1000) + binarize_bin, binarize_bin)

    histograms = session.presentationwise_spike_counts(
        stimulus_presentation_ids=presentations.index.values,  
        bin_edges=time_bins,
        unit_ids=units.index.values)

    binary_spikes = np.array(histograms) # trial, time, unit. use 'histograms.coords' to confirm


    # Given a binary spike matrix, get spike times.
    numTrials, numTimePoints, numUnits = binary_spikes.shape

    SpikeMats = np.empty((numTrials, numUnits), dtype='object')
    for trial in range(numTrials):
        for unit in range(numUnits):
            SpikeMats[trial, unit] = np.where(binary_spikes[trial, :, unit] != 0)[0]

    data_dict = {
        'SpikeMats': SpikeMats,
        'stimIDs': stimIDs,
        'preTrialWindowMS': preTrialWindowMS,
        'postTrialWindowMS': postTrialWindowMS,
        'numTrials': numTrials,
        'numTimePoints': numTimePoints,
        'numUnits': numUnits,
        'region': region
    }
    with open(os.path.join(output_path, f'session_{session_id}.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)