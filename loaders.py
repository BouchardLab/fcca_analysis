import pickle
import itertools
import operator
import numpy as np
import h5py
import glob
import inspect
import warnings
import pdb
import os; os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from tqdm import tqdm
from scipy import io
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.signal import resample,  convolve, get_window
from scipy.ndimage import convolve1d, gaussian_filter1d
from copy import deepcopy
from joblib import Parallel, delayed


start_times = {'indy_20160426_01': 0,
               'indy_20160622_01':1700,
               'indy_20160624_03': 500,
               'indy_20160627_01': 0,
               'indy_20160630_01': 0,
               'indy_20160915_01': 0,
               'indy_20160921_01': 0,
               'indy_20160930_02': 0,
               'indy_20160930_05': 300,
               'indy_20161005_06': 0,
               'indy_20161006_02': 350,
               'indy_20161007_02': 950,
               'indy_20161011_03': 0,
               'indy_20161013_03': 0,
               'indy_20161014_04': 0,
               'indy_20161017_02': 0,
               'indy_20161024_03': 0,
               'indy_20161025_04': 0,
               'indy_20161026_03': 0,
               'indy_20161027_03': 500,
               'indy_20161206_02': 5500,
               'indy_20161207_02': 0,
               'indy_20161212_02': 0,
               'indy_20161220_02': 0,
               'indy_20170123_02': 0,
               'indy_20170124_01': 0,
               'indy_20170127_03': 0,
               'indy_20170131_02': 0,
               'loco_20170210_03':0, 
               'loco_20170213_02':0, 
               'loco_20170214_02':0, 
               'loco_20170215_02':0, 
               'loco_20170216_02': 0, 
               'loco_20170217_02': 0, 
               'loco_20170227_04': 0, 
               'loco_20170228_02': 0, 
               'loco_20170301_05':0, 
               'loco_20170302_02':0}


def measure_straight_dev(trajectory, start, end):
    # Translate to the origin relative to the 1st target location
    trajectory -= start

    # straight line vector
    straight = end - start
    straight_norm = np.linalg.norm(straight)
    straight /= straight_norm

    if straight[0] == 0:
        perp = np.array([1, 0])
    elif straight[1] == 0:
        perp = np.array([0, 1])
    else:
        # Vector orthogonal to the straight line between targets
        x_orth = np.random.uniform(0, 1)
        y_orth = -1 * (straight[0] * x_orth)/straight[1]
        perp = np.array([x_orth, y_orth])
        perp /= np.linalg.norm(perp)
    
    if np.any(np.isnan(perp)):
        pdb.set_trace()
    
    m = straight[1]/straight[0]
    b = 0

    straight_dev = 0
    for j in range(trajectory.shape[0]):
        
        # transition is horizontal
        if m == 0:
            x_int = trajectory[j, 0]
            y_int = straight[1]
        # transition is vertical
        elif np.isnan(m) or np.isinf(m):
            x_int = straight[0]
            y_int = trajectory[j, 1]
        else:
            m1 = -1/m
            b1 = trajectory[j, 1] - m1 * trajectory[j, 0]
            # Find the intersection between the two lines
            x_int = (b - b1)/(m1 - m)
            y_int = m1 * x_int + b1
        
        straight_dev += np.linalg.norm(np.array([x_int - trajectory[j, 0], y_int - trajectory[j, 1]]))

    # Normalize by the length of straight trajectory
    straight_dev /= straight_norm
    return straight_dev

def reach_segment_sabes(dat, start_time=None, data_file=None, keep_high_error=False, err_thresh=0.9):
    print('Reminder that start times depend on the bin size')
    if start_time is None:
        start_time = start_times[data_file]

    target_locs = []
    time_on_target = []
    valid_transition_times = []

    target_diff = np.diff(dat['target'].T)
    # This will yield the last index before the transition
    transition_times = np.sort(np.unique(target_diff.nonzero()[1]))
    #transition_times = target_diff.nonzero()[1]

    # For each transition, make a record of the location, time on target, and transition_vector
    # Throw away those targets that only appear for 1 timestep
    for i, transition_time in enumerate(transition_times):

        # Only lingers at the target for one timestep
        if i < len(transition_times) - 1:
            if np.diff(transition_times)[i] == 1:
                continue

        target_locs.append(dat['target'][transition_time][:])
        valid_transition_times.append(transition_time)
        
    for i, transition_time in enumerate(valid_transition_times):
            
        if i == 0:
            time_on_target.append(transition_time + 1)
        else:
            time_on_target.append(transition_time - valid_transition_times[i - 1] + 1)
            
    target_locs = np.array(target_locs)
    time_on_target = np.array(time_on_target)
    valid_transition_times = np.array(valid_transition_times)

    # Filter out by when motion starts
    if start_time > valid_transition_times[0]:
        init_target_loc = target_locs[valid_transition_times < start_time][-1]
    else:
        init_target_loc = target_locs[0]

    target_locs = target_locs[valid_transition_times > start_time]
    time_on_target = time_on_target[valid_transition_times > start_time]
    valid_transition_times = valid_transition_times[valid_transition_times > start_time]

    # Velocity profiles
    vel = np.diff(dat['behavior'], axis=0)

    target_pairs = []
    for i in range(1, len(target_locs)):
        target_pairs.append((i - 1, i))

    target_error_pairs = np.zeros(len(target_pairs))

    for i in range(len(target_pairs)):
        
    #    time_win = max(min(10, int(0.05 * time_on_target[i])), 2)
        time_win = 2
        
        # Length of time_win just after target switches
        cursor_0 = dat['behavior'][valid_transition_times[target_pairs[i][0]] + 1:\
                                   valid_transition_times[target_pairs[i][0]] + 1 + time_win]
        # Length of time_win just before target switches again
        cursor_1 = dat['behavior'][valid_transition_times[target_pairs[i][1]] - time_win:\
                                   valid_transition_times[target_pairs[i][1]]]

        target_error_pairs[i] = np.max([np.mean(np.linalg.norm(cursor_0 - target_locs[target_pairs[i][0]])),
                                         np.mean(np.linalg.norm(cursor_1 - target_locs[target_pairs[i][1]]))])

    # Thresholding by error threshold (how far from the start and end targets is the reach)
    err_thresh = np.quantile(target_error_pairs, err_thresh)

    # Throw away trajectories with highly erratic velocity profiles
    # (large number of zero crossings in the acceleration)
    n_zeros = np.zeros(len(target_pairs))
    for i in range(len(target_pairs)):
        acc = np.diff(vel[valid_transition_times[target_pairs[i][0]]:\
                          valid_transition_times[target_pairs[i][1]]], axis=0)    
        n_zeros[i] = (np.diff(np.sign(acc)) != 0).sum()

    # Throw away reaches with highest 10 % of target error and > 200 acceleration zero crossings
    # Pair of target corrdinates
    valid_target_pairs = []
    # How long did the reach take
    reach_duration = []
    # Tuple of indices that describes start and end of reach
    transition_times = []
    transition_vectors = []
    nzeros = []

    indices_kept = []

    for i in range(len(target_error_pairs)): 
        # Keep this transition
        if (target_error_pairs[i] < err_thresh and n_zeros[i] < 200) or keep_high_error:
            valid_target_pairs.append((target_locs[target_pairs[i][0]], target_locs[target_pairs[i][1]]))        
            reach_duration.append(time_on_target[target_pairs[i][1]])
            transition_times.append((valid_transition_times[target_pairs[i][0]] + 1,
                                    valid_transition_times[target_pairs[i][1]]))
            transition_vectors.append(target_locs[target_pairs[i][1]] - target_locs[target_pairs[i][0]])
            indices_kept.append(i)
        else: 
            continue


    target_error_pairs = target_error_pairs[np.array(indices_kept)]
    n_zeros = n_zeros[np.array(indices_kept)]

    transition_orientation = np.zeros(len(transition_vectors))
    refvec = np.array([1, 0])
    for i in range(len(transition_vectors)):
        # Normalize
        transvecnorm = transition_vectors[i]/np.linalg.norm(transition_vectors[i])
        dot = transvecnorm @ refvec      # dot product
        det = transvecnorm[0]*refvec[1] - transvecnorm[1]*refvec[0]  # determinant
        transition_orientation[i] = np.arctan2(det, dot)

    # Integrate the area under the trajectory minus the straight line
    straight_dev = np.zeros(len(valid_target_pairs))
    # Operator on a copy of trajectory
    cursor_trajectory = deepcopy(dat['behavior'])
    for i in range(len(valid_target_pairs)):
        
        trajectory = cursor_trajectory[transition_times[i][0]:transition_times[i][1], :]

        straight_dev[i] = measure_straight_dev(trajectory, valid_target_pairs[i][0], 
                                               valid_target_pairs[i][1])

    # Augment dictionary with segmented reaches and their characteristics
    dat['vel'] = vel
    dat['target_pairs'] = valid_target_pairs
    dat['transition_times'] = transition_times
    dat['straight_dev'] = straight_dev
    dat['target_pair_error'] = target_error_pairs
    dat['transition_orientation'] = transition_orientation
    dat['npeaks'] = n_zeros

    return dat

def filter_window(signal, window_name,  window_length=10):
    window = get_window(window_name, window_length)
    signal = convolve1d(signal, window)
    return signal
FILTER_DICT = {'gaussian':gaussian_filter1d, 'none': lambda x, **kwargs: x, 'window': filter_window}


def moving_center(X, n, axis=0):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd

def sinc_filter(X, fc, axis=0):
        
    # Windowed sinc filter
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
    
    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2))

    # Compute Blackman window.
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))

    # Multiply sinc filter by window.
    h = h * w

    # Normalize to get unity gain.
    h = h / np.sum(h)
    return convolve(X, h)        

def window_spike_array(spike_times, tstart, tend):
    windowed_spike_times = np.zeros(spike_times.shape, dtype=object)

    for i in range(spike_times.shape[0]):
        for j in range(spike_times.shape[1]):
            wst, _ = window_spikes(spike_times[i, j], tstart[i], tend[i])
            windowed_spike_times[i, j] = wst

    return windowed_spike_times

def window_spikes(spike_times, tstart, tend, start_idx=0):

    spike_times = spike_times[start_idx:]
    spike_times[spike_times > tstart]

    if len(spike_times) > 0:
        start_idx = np.argmax(spike_times > tstart)
        end_idx = np.argmin(spike_times < tend)

        windowed_spike_times = spike_times[start_idx:end_idx]

        # Offset spike_times to start at 0
        if windowed_spike_times.size > 0:
                windowed_spike_times -= tstart

        return windowed_spike_times, end_idx - 1
    else:
        return np.array([]), start_idx

def align_behavior(x, T, bin_width):
    
    bins = np.linspace(0, T, int(T//bin_width))
    bin_centers = bins + (bins[1] - bins[0])/2
    bin_centers = bin_centers[:-1]
    xaligned = np.zeros((bin_centers.size, x.shape[-1]))
    
    for j in range(x.shape[-1]):
        interpolator = interp1d(np.linspace(0, T, x[:, j].size), x[:, j])
        xaligned[:, j] = interpolator(bin_centers)

    return xaligned

def align_peanut_behavior(t, x, bins):
    # Offset to 0
    t -= t[0]
    bin_centers = bins + (bins[1] - bins[0])/2
    bin_centers = bin_centers[:-1]
    interpolator = interp1d(t, x, axis=0)
    xaligned = interpolator(bin_centers)
    return xaligned, bin_centers

# spike_times: (n_trial, n_neurons)
#  trial threshold: If we require a spike threshold, trial threshold = 1 requires 
#  the spike threshold to hold for the neuron for all trials. 0 would mean no trials

# Need to (1) speed this guy up, (2) make sure filtering is doing the right thing
# (3) remove parisitic memory usage

def postprocess_spikes(spike_times, T, bin_width, boxcox, filter_fn, filter_kwargs,
                       spike_threshold=0, trial_threshold=1, high_pass=False, return_unit_filter=False):

    # Trials are of different duration
    if np.isscalar(T):
        ragged_trials = False
    else:
        ragged_trials = True

    # Discretize time over bins
    if ragged_trials:
        bins = []
        for i in range(len(T)):
            bins.append(np.linspace(0, T[i], int(T[i]//bin_width)))
        bins = np.array(bins, dtype=object)
        spike_rates = np.zeros((spike_times.shape[0], spike_times.shape[1]), dtype=object)
    else:
        bins = np.linspace(0, T, int(T//bin_width))
        spike_rates = np.zeros((spike_times.shape[0], spike_times.shape[1], bins.size - 1,))    

    # Did the trial/unit have enough spikes?
    insufficient_spikes = np.zeros(spike_times.shape)
    #print('Processing spikes')
    #for i in tqdm(range(spike_times.shape[0])):
    for i in range(spike_times.shape[0]):
        for j in range(spike_times.shape[1]):    

            # Ignore this trial/unit combo
            if np.any(np.isnan(spike_times[i, j])):
                insufficient_spikes[i, j] = 1          

            if ragged_trials:
                spike_counts = np.histogram(spike_times[i, j], bins=np.squeeze(bins[i]))[0]    
            else:
                spike_counts = np.histogram(spike_times[i, j], bins=bins)[0]

            if spike_threshold is not None:
                if np.sum(spike_counts) <= spike_threshold:
                    insufficient_spikes[i, j] = 1

            # Apply a boxcox transformation
            if boxcox is not None:
                spike_counts = np.array([(np.power(spike_count, boxcox) - 1)/boxcox 
                                         for spike_count in spike_counts])

            # Filter only if we have to, otherwise vectorize the process
            if ragged_trials:
                # Filter the resulting spike counts
                spike_rates_ = FILTER_DICT[filter_fn](spike_counts.astype(float), **filter_kwargs)
                # High pass to remove long term trends (needed for sabes data)
                if high_pass:
                    spike_rates_ = moving_center(spike_rates_, 600)
            else:
                spike_rates_ = spike_counts
            spike_rates[i, j] = spike_rates_

    # Filter out bad units
    sufficient_spikes = np.arange(spike_times.shape[1])[np.sum(insufficient_spikes, axis=0) < \
                                                        (1 - (trial_threshold -1e-3)) * spike_times.shape[0]]
    spike_rates = spike_rates[:, list(sufficient_spikes)]

    # Transpose so time is along the the second 'axis'
    if ragged_trials:
        spike_rates = [np.array([spike_rates[i, j] for j in range(spike_rates.shape[1])]).T for i in range(spike_rates.shape[0])]
    else:
        # Filter the resulting spike counts
        spike_rates = FILTER_DICT[filter_fn](spike_rates, **filter_kwargs)
        # High pass to remove long term trends (needed for sabes data)
        if high_pass:
            spike_rates = moving_center(spike_rates, 600, axis=-1)

        spike_rates = np.transpose(spike_rates, (0, 2, 1))

    if return_unit_filter:
        return spike_rates, sufficient_spikes
    else:
        return spike_rates

def load_sabes_trialized(filename, min_length=6, **kwargs):

    # start time is handled in reach_segment_sabes, so do not prematurely truncate
    kwargs['truncate_start'] = False
    kwargs['segment'] = False
    # Load the data
    dat = load_sabes(filename, **kwargs)
    # Trialize
    dat_segment = reach_segment_sabes(dat, data_file=filename.split('/')[-1].split('.mat')[0])
    # Modfiy the spike rates and behavior entries according to the segmentation
    spike_rates = dat['spike_rates'].squeeze()
    spike_rates_trialized = [spike_rates[tt[0]:tt[1], :] 
                             for tt in dat_segment['transition_times']
                             if tt[1] - tt[0] > min_length]
    behavior = dat['behavior'].squeeze()
    behavior_trialized = [behavior[tt[0]:tt[1], :] for tt in dat_segment['transition_times']]
    dat['spike_rates'] = np.array(spike_rates_trialized, dtype=object)
    dat['behavior'] = np.array(behavior_trialized, dtype=object)
    return dat

def load_sabes(filename, bin_width=50, boxcox=0.5, filter_fn='none', filter_kwargs={}, spike_threshold=100,
               std_behavior=False, region='M1', high_pass=True, segment=False, return_wf=False, 
               subset=None, truncate_start=False, **kwargs):
    print('Start loading Sabes data...')
    # Convert bin width to s
    bin_width /= 1000

    # Load MATLAB file
    # Avoid random OS errors
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']

        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        dat = {}

        if region == 'M1':
            indices = M1_indices
        elif region == 'S1':
            indices = S1_indices
        elif region == 'both':
            indices = list(range(n_channels))

        # Perform binning
        n_channels = len(indices)
        n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
        n_units = n_channels * n_sorted_units
        max_t = t[-1]

        spike_times = np.zeros((n_sorted_units - 1, len(indices))).astype(object)
        if return_wf:
            wf = np.zeros((n_sorted_units - 1, len(indices))).astype(object)

        for i, chan_idx in enumerate(indices):
            for unit_idx in range(1, n_sorted_units): # ignore hash
                spike_times_ = f[f["spikes"][unit_idx, chan_idx]][()]
                # Ignore this case (no data)
                if spike_times_.shape == (2,):
                    spike_times[unit_idx - 1, i] = np.nan
                else:
                    # offset spike times
                    spike_times[unit_idx - 1, i] = spike_times_[0, :] - t[0]
    
                if return_wf:
                    wf[unit_idx - 1, i] = f[f['wf'][unit_idx, chan_idx]][()].T

        # Reshape into format (ntrials, units)
        spike_times = spike_times.reshape((1, -1))
        if return_wf:
            wf = wf.reshape((1, -1))
        # Total length of the time series
        T = t[-1] - t[0]
        if return_wf:
            spike_rates, sufficient_spikes = postprocess_spikes(spike_times, T, bin_width, boxcox,
                                                                filter_fn, filter_kwargs, spike_threshold, high_pass=high_pass,
                                                                return_unit_filter=True)
            
            wf = wf[:, list(sufficient_spikes)]               
            dat['wf'] = wf
        else:
            spike_rates = postprocess_spikes(spike_times, T, bin_width, boxcox,
                                             filter_fn, filter_kwargs, spike_threshold, high_pass=high_pass)
        dat['spike_rates'] = spike_rates 

        # Get cursor position
        cursor_pos = f["cursor_pos"][:].T
        cursor_interp = align_behavior(cursor_pos, T, bin_width)
        if std_behavior:
            cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
            cursor_interp /= cursor_interp.std(axis=0, keepdims=True)

        dat["behavior"] = cursor_interp

        # Target position
        target_pos = f["target_pos"][:].T
        target_interp = align_behavior(target_pos, T, bin_width)
        # cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
        # cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        dat['target'] = target_interp

        dat['time'] = np.squeeze(align_behavior(t[:, np.newaxis], T, bin_width))

        # Pass through reach_segment_sabes and re-assign the behavior and spike_rates keys to the segmented versions 
        if segment:
            dat = reach_segment_sabes(dat, data_file=filename.split('/')[-1].split('.mat')[0])

            # Ensure we have somewhat long trajectories
            # T = 30
            # t = np.array([t_[1] - t_[0] for t_ in dat['transition_times']])
            # valid_transitions = np.arange(t.size)[t >= T]
            valid_transitions = np.arange(len(dat['transition_times']))
            spike_rates = np.array([dat['spike_rates'][0, dat['transition_times'][idx][0]:dat['transition_times'][idx][1]]
                                    for idx in valid_transitions])
            behavior = np.array([dat['behavior'][dat['transition_times'][idx][0]:dat['transition_times'][idx][1]]
                                 for idx in valid_transitions])

            dat['spike_rates'] = spike_rates
            dat['behavior'] = behavior
        
        if truncate_start:
            dat['spike_rates'] = dat['spike_rates'][:, start_times[filename.split('/')[-1].split('.mat')[0]]:]
            dat['behavior'] = dat['behavior'][start_times[filename.split('/')[-1].split('.mat')[0]]:]
        # Select a subset of neurons only
        if subset is not None:
            key = filename.split('/')[-1]
            if key not in subset:
                key = key.split('.mat')[0]
            dat['spike_rates'] = dat['spike_rates'][..., subset[key]]
        return dat

def load_sabes_wf(filename, spike_threshold=100, region='M1'):

    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']

        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        dat = {}

        if region == 'M1':
            indices = M1_indices
        elif region == 'S1':
            indices = S1_indices
        elif region == 'both':
            indices = list(range(n_channels))

        # Perform binning
        n_channels = len(indices)
        n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
        n_units = n_channels * n_sorted_units
        max_t = t[-1]

        spike_times = np.zeros((n_sorted_units - 1, len(indices))).astype(object)
        wf = np.zeros((n_sorted_units - 1, len(indices))).astype(object)

        for i, chan_idx in enumerate(indices):
            for unit_idx in range(1, n_sorted_units): # ignore hash
                spike_times_ = f[f["spikes"][unit_idx, chan_idx]][()]
                # Ignore this case (no data)
                if spike_times_.shape == (2,):
                    spike_times[unit_idx - 1, i] = np.nan
                else:
                    # offset spike times
                    spike_times[unit_idx - 1, i] = spike_times_[0, :] - t[0]

                wf[unit_idx - 1, i] = f[f['wf'][unit_idx, chan_idx]][()].T
            
        # # Reshape into format (ntrials, units)
        spike_times = spike_times.reshape((1, -1))
        wf = wf.reshape((1, -1))

        # # Apply spike threshold
        sizes = np.array([[1 if np.isscalar(spike_times[i, j]) else spike_times[i, j].size for j in range(spike_times.shape[1])]
                          for i in range(spike_times.shape[0])])

        sufficient_spikes = np.zeros(spike_times.shape).astype(np.bool_)
        for i in range(spike_times.shape[0]):
            for j in range(spike_times.shape[1]):
                if spike_threshold is not None:
                    if sizes[i, j] > spike_threshold:
                        sufficient_spikes[i, j] = 1
        
        wf = wf[sufficient_spikes]
        return wf


def load_peanut_across_epochs(fpath, epochs, spike_threshold, **loader_kwargs):

    dat_allepochs = {}
    dat_per_epoch = []

    unit_ids = []

    for epoch in epochs:
        dat = load_peanut(fpath, epoch, spike_threshold, **loader_kwargs)
        unit_ids.append(set(dat['unit_ids']))
        dat_per_epoch.append(dat)

    unit_id_intersection = unit_ids[0]
    for i in range(1, len(epochs)):
        unit_id_intersection.intersection(unit_ids[i])

    for i, epoch in enumerate(epochs):
        dat = dat_per_epoch[i]
        unit_idxs = np.isin(dat['unit_ids'], np.array(list(unit_id_intersection)).astype(int)) 

def load_peanut(fpath, epoch, spike_threshold, bin_width=25, boxcox=0.5,
                filter_fn='none', speed_threshold=4, region='HPc', filter_kwargs={}):
    '''
        Parameters:
            fpath: str
                 path to file
            epoch: list of ints
                which epochs (session) to load. The rat is sleeping during odd numbered epochs
            spike_threshold: int
                throw away neurons that spike less than the threshold during the epoch
            bin_width:  float 
                Bin width for binning spikes. Note the behavior is sampled at 25ms
            boxcox: float or None
                Apply boxcox transformation
            filter_fn: str
                Check filter_dict
            filter_kwargs
                keyword arguments for filter_fn
    '''

    data = pickle.load(open(fpath, 'rb'))
    dict_ = data['peanut_day14_epoch%d' % epoch]
    
    # Collect single units located in hippocampus

    HPc_probes = [key for key, value in dict_['identification']['nt_brain_region_dict'].items()
                  if value in ['HPc', 'HPC']]

    OFC_probes = [key for key, value in dict_['identification']['nt_brain_region_dict'].items()
                  if value == 'OFC']

    if region in ['HPc', 'HPC']:
        probes = HPc_probes
    elif region == 'OFC':
        probes = OFC_probes
    elif region == 'both':
        probes = list(set(HPc_probes).union(set(OFC_probes)))

    spike_times = []
    unit_ids = []
    for probe in dict_['spike_times'].keys():
        probe_id = probe.split('_')[-1]
        if probe_id in probes:
            for unit, times in dict_['spike_times'][probe].items():
                spike_times.append(list(times))
                unit_ids.append((probe_id, unit))
        else:
            continue


    # sort spike times
    spike_times = [list(np.sort(times)) for times in spike_times]

    # Apply spike threshold

    spike_threshold_filter = [idx for idx in range(len(spike_times))
                              if len(spike_times[idx]) > spike_threshold]
    spike_times = np.array(spike_times, dtype=object)
    spike_times = spike_times[spike_threshold_filter]
    unit_ids = np.array(unit_ids)[spike_threshold_filter]

    t = dict_['position_df']['time'].values
    T = t[-1] - t[0] 
    # Convert bin width to s
    bin_width = bin_width/1000
    
    # covnert smoothin bandwidth to indices
    if filter_fn == 'gaussian':
        filter_kwargs['sigma'] /= bin_width
        filter_kwargs['sigma'] = min(1, filter_kwargs['sigma'])
    
    bins = np.linspace(0, T, int(T//bin_width))

    spike_rates = np.zeros((bins.size - 1, len(spike_times)))
    for i in range(len(spike_times)):
        # translate to 0
        spike_times[i] -= t[0]
        
        spike_counts = np.histogram(spike_times[i], bins=bins)[0]
        if boxcox is not None:
            spike_counts = np.array([(np.power(spike_count, boxcox) - 1)/boxcox
                                     for spike_count in spike_counts])
        spike_rates_ = FILTER_DICT[filter_fn](spike_counts.astype(float), **filter_kwargs)
        
        spike_rates[:, i] = spike_rates_
    
    # Align behavior with the binned spike rates
    pos_linear = dict_['position_df']['position_linear'].values
    pos_xy = np.array([dict_['position_df']['x-loess'], dict_['position_df']['y-loess']]).T
    pos_linear, taligned = align_peanut_behavior(t, pos_linear, bins)
    pos_xy, _ = align_peanut_behavior(t, pos_xy, bins)
    
    dat = {}
    dat['unit_ids'] = unit_ids
    # Apply movement threshold
    if speed_threshold is not None:
        vel = np.divide(np.diff(pos_linear), np.diff(taligned))
        # trim off first index to match lengths
        spike_rates = spike_rates[1:, ...]
        pos_linear = pos_linear[1:, ...]
        pos_xy = pos_xy[1:, ...]

        spike_rates = spike_rates[np.abs(vel) > speed_threshold]

        pos_linear = pos_linear[np.abs(vel) > speed_threshold]
        pos_xy = pos_xy[np.abs(vel) > speed_threshold]

    dat['unit_ids'] = unit_ids
    dat['spike_rates'] = spike_rates
    dat['behavior'] = pos_xy
    dat['behavior_linear'] = pos_linear[:, np.newaxis]
    dat['time'] = taligned
    return dat

##### Peanut Segmentation #####
def segment_peanut(dat, loc_file, epoch, box_size=20, start_index=0, return_maze_points=False):

    with open(loc_file, 'rb') as f:
        ldict = pickle.load(f)
        
    edgenames = ldict['peanut_day14_epoch2']['track_graph']['edges_ordered_list']
    nodes = ldict['peanut_day14_epoch%d' % epoch]['track_graph']['nodes']
    for key, value in nodes.items():
        nodes[key] = (value['x'], value['y'])
    endpoints = []
    lengths = []
    for edgename in edgenames:
        endpoints.append(ldict['peanut_day14_epoch%d' % epoch]['track_graph']['edges'][edgename]['endpoints'])
        lengths.append(ldict['peanut_day14_epoch%d' % epoch]['track_graph']['edges'][edgename]['length'])
        
    # pos = np.array([ldict['peanut_day14_epoch%d' % epoch]['position_input']['position_x'],
    #             ldict['peanut_day14_epoch%d' % epoch]['position_input']['position_y']]).T
    pos = dat['behavior']
    if epoch in [2, 6, 10, 14]:
        transition1 = find_transitions(pos, nodes, 'handle_well', 'left_well', 
                                                   ignore=['center_maze', 'left_corner'], box_size=box_size, start_index=start_index)
        transition2 = find_transitions(pos, nodes, 'handle_well', 'right_well',
                                                   ignore=['center_maze', 'right_corner'], box_size=box_size, start_index=start_index)
    elif epoch in [4, 8, 12, 16]:
        transition1 = find_transitions(pos, nodes, 'center_well', 'left_well', 
                                                   ignore=['center_maze', 'left_corner'], box_size=box_size, start_index=start_index)
        transition2 = find_transitions(pos, nodes, 'center_well', 'right_well',
                                                   ignore=['center_maze', 'right_corner'], box_size=box_size, start_index=start_index)
    if return_maze_points:
        return transition1, transition2, nodes, endpoints
    else:
        return transition1, transition2

def in_box(pos, node, box_size):
    box_points = [np.array(node) + box_size/2 * np.array([1, 1]), # Top right
                  np.array(node) + box_size/2 * np.array([1, -1]), # Bottom right
                  np.array(node) + box_size/2 * np.array([-1, 1]), # Top left
                  np.array(node) + box_size/2 * np.array([-1, -1])] # Bottom left

    in_xlim = np.bitwise_and(pos[:, 0] > box_points[-1][0], 
                             pos[:, 0] < box_points[0][0])
    in_ylim = np.bitwise_and(pos[:, 1] > box_points[-1][1], 
                             pos[:, 1] < box_points[0][1])    
    return np.bitwise_and(in_xlim, in_ylim)
    
def find_transitions(pos, nodes, start_node, end_node, ignore=['center_maze'],
                     box_size=20, start_index=1000):
    pos = pos[start_index:]
    
    in_node_boxes = {}
    for key, value in nodes.items():
        in_node_boxes[key] = in_box(pos, value, box_size)
        
    in_node_boxes_windows = {}
    for k in in_node_boxes.keys():
        in_node_boxes_windows[k] = [[i for i,value in it] 
                                    for key,it in 
                                    itertools.groupby(enumerate(in_node_boxes[k]), key=operator.itemgetter(True)) 
                                    if key != 0]

    # For each window of time that the rat is in the start node box, find which box it goes to next. If this
    # box matches the end_node, then add the intervening indices to the list of transitions
    transitions = []
    for start_windows in in_node_boxes_windows[start_node]:
        next_box_times = {}
        
        # When does the rat leave the start_node
        t0 = start_windows[-1]
        for key, windows in in_node_boxes_windows.items():
            window_times = np.array([time for window in windows for time in window])
            # what is the first time after t0 that the rat enters this node/box
            valid_window_times = window_times[window_times > t0]
            if len(valid_window_times) > 0:
                next_box_times[key] = window_times[window_times > t0][0]
            else:
                next_box_times[key] = np.inf

        # Order the well names by next_box_times
        node_names = list(next_box_times.keys())
        node_times = list(next_box_times.values())
        
        
        node_order = np.argsort(node_times)
        idx = 0
        # Find the first node that is not the start_node and is not in the list of nodes to ignore
        while (node_names[node_order[idx]] in ignore) or (node_names[node_order[idx]] == start_node):
            idx += 1

        if node_names[node_order[idx]] == end_node:
            # Make sure to translate by the start index
            transitions.append(np.arange(t0, node_times[node_order[idx]]) + start_index)
            
    return transitions


# # Avoids use of AllenSDK
def load_AllenVC(data_path, region="VISp", bin_width=25, 
                 preTrialWindowMS=50, postTrialWindowMS=100, 
                 boxcox=0.5, filter_fn='none', filter_kwargs={}, 
                 spike_threshold=None, trial_threshold=0):

    # Loads one session at a time
    
    # ------------------------------- Check if these params have already been applied/loaded first, or load new ::
    
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)    
    arg_dict = {arg: values[arg] for arg in args}
    arg_dict['filter_kwargs'] = make_hashable(arg_dict['filter_kwargs'])    
    arg_tuple = tuple(sorted(arg_dict.items()))
    
    DataFolderPath = os.path.dirname(os.path.dirname(data_path))
    preload_dict_path = DataFolderPath + '/preloaded/preloadDict.pickle'
    with open(preload_dict_path, 'rb') as file:
        preloadDict = pickle.load(file)

    for args in preloadDict.keys():
        if args == arg_tuple:
            print("Preloading data...")
            preloadID = preloadDict[arg_tuple]
            loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
            with open(loaded_data_path, 'rb') as file:
                dat = pickle.load(file)
            return dat

    # ------------------------------- Otherwise, load the data from preloaded_spikes    
    session_id = int(os.path.splitext(os.path.basename(data_path))[0].split('_')[1])
    preloaded_spikes_path = '/'.join(os.path.dirname(preload_dict_path).split('/')[:-1]) \
        + f"/preloaded_spikes/session_{session_id}.pickle"
    with open(preloaded_spikes_path, 'rb') as file:
        data_dict = pickle.load(file)
    SpikeMats = data_dict['SpikeMats']    
    stimIDs = data_dict['stimIDs']
    numTimePoints = data_dict['numTimePoints']

    preTrialWindowMS = data_dict['preTrialWindowMS']
    postTrialWindowMS = data_dict['postTrialWindowMS']
    region_ = data_dict['region']
    assert(region_ == region)

    # ------------------------------- Filter spikes    
    T = numTimePoints # units of ms duration of a trial (here, includes pre- and post- windows)
    spike_rates = postprocess_spikes(SpikeMats, T, bin_width, boxcox, filter_fn, dict(filter_kwargs), spike_threshold=spike_threshold, trial_threshold=trial_threshold)
    
    
    
    dat = {}
    dat["spike_rates"] = spike_rates
    dat["behavior"] = stimIDs
    dat["preTrialWindow"] = preTrialWindowMS
    dat["postTrialWindow"] = postTrialWindowMS
    dat["spike_times"] = SpikeMats



    # ------------------------------- Save this data run for the future

    # Assign an ID to this loader call
    if not preloadDict: preloadID = 0
    else: preloadID = max(list(preloadDict.values())) + 1
    preloadDict[arg_tuple] = preloadID 

    # Save the preload dict and the actual data
    with open(preload_dict_path, 'wb') as file:
        pickle.dump(preloadDict, file)

    loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
    with open(loaded_data_path, 'wb') as file:
        pickle.dump(dat, file)

    
    return dat


def load_AllenVC_allensdk(data_path, region="VISp", bin_width=25, preTrialWindowMS=50, postTrialWindowMS=100, boxcox=0.5, filter_fn='none', filter_kwargs={}, spike_threshold=None, trial_threshold=0):

    # Loads one session at a time
    
    # ------------------------------- Check if these params have already been applied/loaded first, or load new ::
    
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)    
    arg_dict = {arg: values[arg] for arg in args}
    arg_dict['filter_kwargs'] = make_hashable(arg_dict['filter_kwargs'])    
    arg_tuple = tuple(sorted(arg_dict.items()))
    
    DataFolderPath = os.path.dirname(os.path.dirname(data_path))
    preload_dict_path = DataFolderPath + '/preloaded/preloadDict.pickle'
    with open(preload_dict_path, 'rb') as file:
        preloadDict = pickle.load(file)

    for args in preloadDict.keys():
        if args == arg_tuple:
            print("Preloading data...")
            preloadID = preloadDict[arg_tuple]
            loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
            with open(loaded_data_path, 'rb') as file:
                dat = pickle.load(file)
            return dat


    # ------------------------------- Otherwise, load the data "fresh"    
    print("Begin Loading Data Fresh ...")
    

    # Get Allen structures for loading data
    manifest_path = os.path.join(DataFolderPath, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    session_id = int(os.path.splitext(os.path.basename(data_path))[0].split('_')[1])

    
    # For all session info, including regions, stimulus names, unit count, etc. see: session.metadata
    warnings.filterwarnings("ignore", category=UserWarning)
    session = cache.get_session_data(session_id)
        
    units = session.units[session.units["ecephys_structure_acronym"] == region]
    if units.empty: return {} # Check that this region is in this session and has units

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
    
    # SpikeMats reports for each (trial, unit) the time in ms of a spike
    # RELATIVE TO "preTrialWindow" seconds before the trial starts, and until "postTrialWindow" seconds after the trial starts
    
    
    T = numTimePoints # units of ms duration of a trial (here, includes pre- and post- windows)
    spike_rates = postprocess_spikes(SpikeMats, T, bin_width, boxcox, filter_fn, dict(filter_kwargs), spike_threshold=spike_threshold, trial_threshold=trial_threshold)
    
    
    
    dat = {}
    dat["spike_rates"] = spike_rates
    dat["behavior"] = stimIDs
    dat["preTrialWindow"] = preTrialWindowMS
    dat["postTrialWindow"] = postTrialWindowMS
    dat["spike_times"] = SpikeMats



    # ------------------------------- Save this data run for the future

    # Assign an ID to this loader call
    if not preloadDict: preloadID = 0
    else: preloadID = max(list(preloadDict.values())) + 1
    preloadDict[arg_tuple] = preloadID 

    # Save the preload dict and the actual data
    with open(preload_dict_path, 'wb') as file:
        pickle.dump(preloadDict, file)

    loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
    with open(loaded_data_path, 'wb') as file:
        pickle.dump(dat, file)

    
    return dat


def make_hashable(d):
    """ Recursively convert a dictionary into a hashable type (tuples of tuples). """
    if isinstance(d, dict):
        return tuple((key, make_hashable(value)) for key, value in sorted(d.items()))
    elif isinstance(d, list):
        return tuple(make_hashable(value) for value in d)
    else:
        return d