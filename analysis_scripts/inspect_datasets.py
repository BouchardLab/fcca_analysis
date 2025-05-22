import pdb
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
import itertools

from region_select import *
from config import PATH_DICT

regions = ['M1_psid', 'S1_psid', 'HPC_peanut', 'VISp']

# Scratch script to inspect datasets, get unit counts, etc...
for region in regions:
    df, session_key = load_decoding_df(region, **loader_kwargs[region])

    sessions = np.unique(df[session_key])
    unit_counts = []
    for session in sessions:
        df_ = apply_df_filters(df, **{session_key: session})
        unit_count = df_.iloc[0]['coef'].shape[0]
        unit_counts.append(unit_count)
        print(f'Session {session}: {unit_count} units')

    print(f'Region {region}: {np.mean(unit_counts)} units on average')
