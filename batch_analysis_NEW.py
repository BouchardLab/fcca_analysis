# ------------------------------------------------------------------ Public Packages
import os
import argparse
import pickle
import glob
import itertools
import numpy as np

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from schwimmbad import MPIPool, SerialPool

# ------------------------------------------------------------------ Custom Built Packages
from mpi_loaders import mpi_load_shenoy
from dimreduc_wrappers import PCA_wrapper, NoDimreduc, RandomDimreduc
from loaders import (load_sabes, load_shenoy, load_peanut,  load_cv, load_shenoy_large, load_sabes_trialized, load_franklab_new, load_tsao, load_AllenVC, load_organoids, load_preprocessed)
from decoders import (lr_decoder, rrlr_decoder, logreg, lr_residual_decoder, svm_decoder, psid_decoder, rrglm_decoder)
try:
    from FCCA.fcca import FCCA as LQGCA
except:
    from FCCA_private.FCCA.fcca import LQGComponentsAnalysis as LQGCA

# ------------------------------------------------------------------ Reference Dictionaries
LOADER_DICT = {'sabes': load_sabes, 'shenoy': mpi_load_shenoy, 'peanut': load_peanut, 'cv':load_cv, 'preprocessed': load_preprocessed,
                'mc_maze':load_shenoy_large, 'sabes_trialized': load_sabes_trialized,
                'franklab_new':load_franklab_new, 'tsao':load_tsao, 'AllenVC':load_AllenVC, 'load_organoids':load_organoids}
DECODER_DICT = {'lr': lr_decoder, 'lr_residual': lr_residual_decoder,
                'svm':svm_decoder, 'psid':psid_decoder, 'rrlr': rrlr_decoder, 'logreg':logreg,
                'rrlogreg':rrglm_decoder}
DIMREDUC_DICT = {'PCA': PCA_wrapper, 'LQGCA': LQGCA, 'None':NoDimreduc, 'Random': RandomDimreduc}


# ------------------------------------------------------------------ Parallelization Functions
def prune_tasks(tasks, results_folder, task_format):
    # If the results file exists, there is nothing left to do
    if os.path.exists('%s.dat' % results_folder):
        return []

    completed_files = glob.glob('%s/*.dat' % results_folder)
    param_tuples = []
    for completed_file in completed_files:
        dim = int(completed_file.split('dim_')[1].split('_')[0])
        fold_idx = int(completed_file.split('fold_')[1].split('.dat')[0])
        param_tuples.append((dim, fold_idx))            

    to_do = []
    for task in tasks:
        if task_format == 'dimreduc':
            train_test_tuple, dim, _, _, _ = task
            fold_idx, _, _ = train_test_tuple

            if (dim, fold_idx) not in param_tuples:
                to_do.append(task)

        elif task_format == 'decoding':
            dim, fold_idx, _, _, _ = task

            if (dim, fold_idx) not in param_tuples:
                to_do.append(task)


    return to_do

def consolidate(results_folder, results_file, comm):
    # Only rank 0 (or serial mode) should consolidate files
    if comm is None or comm.rank == 0:
        data_files = glob.glob(f"{results_folder}/*.dat")
        results_dict_list = []

        for data_file in data_files:
            try:
                with open(data_file, "rb") as f:
                    results_dict_list.append(pickle.load(f))
            except:
                os.remove(data_file)  # Delete corrupted file
                return

        with open(results_file, "wb") as f:
            pickle.dump(results_dict_list, f)

class PoolWorker():

    # Initialize the worker with the data so it does not have to be broadcast by pool.map
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

  
    def dimreduc(self, task_tuple):

        task_tuple, comm = task_tuple if len(task_tuple) == 2 else (task_tuple, None)            

        train_test_tuple, dim, method, method_args, results_folder = task_tuple
        fold_idx, train_idxs, test_idxs = train_test_tuple
        print('Dim: %d, Fold idx: %d' % (dim, fold_idx))

        
        X = globals()['X'] # X is either of shape (n_time, n_dof) or (n_trials,) 

        # Check if more dimensions are rquested than are in the data 
        dim_error = X.shape[1] <= dim if np.ndim(X) == 2 else X[0].shape[1] <= dim
        if dim_error:
            print(f"DIM ERROR OCCURRED: Dim={dim}, X.shape={getattr(X, 'shape', 'Unknown')}")
            results_dict = {}
             
        else:
            X_train = X[train_idxs, ...]

            if X.dtype == 'object':  # subtract the cross condition mean
                cross_cond_mean = np.mean([np.mean(x_, axis=0) for x_ in X_train], axis=0)      
                X_train = [x_ - cross_cond_mean for x_ in X_train]
            else:  # Save memory
                X_train -= np.concatenate(X_train).mean(axis=0, keepdims=True)

            # Fit Dimreduc Model
            dimreducmodel = DIMREDUC_DICT[method](d=dim, **method_args)
            dimreducmodel.fit(X_train)
            coef = dimreducmodel.coef_
            score = dimreducmodel.score()
            
            # Organize results in a dictionary structure
            results_dict = {}
            results_dict['dim'] = dim
            results_dict['fold_idx'] = fold_idx
            results_dict['train_idxs'] = train_idxs
            results_dict['test_idxs'] = test_idxs
            results_dict['dimreduc_method'] = method
            results_dict['dimreduc_args'] = method_args
            results_dict['coef'] = coef
            results_dict['score'] = score

        # Write to file, will later be concatenated by the main process
        file_name = 'dim_%d_fold_%d.dat' % (dim, fold_idx)
        with open('%s/%s' % (results_folder, file_name), 'wb') as f:
            f.write(pickle.dumps(results_dict))
        
        return 0 # Cannot return None or else schwimmbad with hang 

    def decoding(self, task_tuple):

        task_tuple, comm = task_tuple if len(task_tuple) == 2 else (task_tuple, None)              

        dim_val, fold_idx, dimreduc_results, decoder, results_folder = task_tuple
        print('Working on %d, %d' % (dim_val, fold_idx))
        coef_ = dimreduc_results['coef']


        X = globals()['X']
        Y = globals()['Y']
        
        # Project the (train and test) data onto the subspace and train and score the requested decoder
        train_idxs = dimreduc_results['train_idxs']
        test_idxs = dimreduc_results['test_idxs']
        Ytrain = Y[train_idxs]
        Ytest = Y[test_idxs]
        Xtrain = X[train_idxs]
        Xtest = X[test_idxs]

        if dim_val <= 0:
            dim_val = Xtrain.shape[-1] if np.ndim(Xtrain) == 2 else Xtrain[0].shape[-1]
            

        # If the coefficient is 3-D, need to do decoding for each (leading) dimension
        if coef_.ndim == 3:
            results_dict_list = []
            for cf in coef_:
                try:
                    cf = cf[:, :dim_val] if dim_val > 1 else cf[:, np.newaxis]
                except:
                    raise ValueError("Invalid shape adjustment for cf.")

                Xtrain_ = Xtrain @ cf if np.ndim(Xtrain) == 2 else [xx @ cf for xx in Xtrain]
                Xtest_ = Xtest @ cf if np.ndim(Xtest) == 2 else [xx @ cf for xx in Xtest]
                Ytrain_, Ytest_ = list(Ytrain), list(Ytest)
                
                results = DECODER_DICT[decoder['method']](Xtest_, Xtrain_, Ytest, Ytrain, **decoder['args'])
                results_dict = {**dimreduc_results[dimreduc_idx], **results}
                results_dict.update({'dim': dim_val, 'fold_idx': fold_idx, 'decoder': decoder['method'], 'decoder_args': decoder['args'] })
                results_dict_list.append(results_dict)
                
            with open('%s/dim_%d_fold_%d.dat' % (results_folder, dim_val, fold_idx), 'wb') as f:
                f.write(pickle.dumps(results_dict_list))
                
        else:
            # Chop off superfluous dimensions (sometimes PCA fits returned all columns of the projection)
            try:
                coef_ = coef_[:, :dim_val] if dim_val > 1 else coef_[:, np.newaxis]
            except:
                raise ValueError("Invalid shape adjustment for coef_.")

            # Apply transformation
            Xtrain = Xtrain @ coef_ if np.ndim(Xtrain) == 2 else [xx @ coef_ for xx in Xtrain]
            Xtest = Xtest @ coef_ if np.ndim(Xtest) == 2 else [xx @ coef_ for xx in Xtest]
            Ytrain, Ytest = list(Ytrain), list(Ytest)  # Convert to list if needed

            results = DECODER_DICT[decoder['method']](Xtest, Xtrain, Ytest, Ytrain, **decoder['args'])
            results_dict = {**dimreduc_results[dimreduc_idx], **results}
            results_dict.update({'dim': dim_val, 'fold_idx': fold_idx, 'decoder': decoder['method'], 'decoder_args': decoder['args'] })
                    
            with open('%s/dim_%d_fold_%d.dat' % (results_folder, dim_val, fold_idx), 'wb') as f:
                f.write(pickle.dumps(results_dict))  
            
        return 0 # Cannot return None or else schwimmbad with hang 
          
# ------------------------------------------------------------------ Main Features 

def load_data(loader, data_file, loader_args, comm, broadcast_behavior=False):

    # Load data on rank 0 (or in serial mode)
    if comm is None or comm.rank == 0:
        dat = LOADER_DICT[loader](data_file, **loader_args)
        spike_rates = np.squeeze(dat['spike_rates'])

        # Enforce that trialized data is formatted as a list of trials or ensure contiguous storage for better performance
        if isinstance(spike_rates, np.ndarray) and spike_rates.ndim == 3:
            spike_rates = np.array([s for s in spike_rates], dtype=object)

        elif not isinstance(spike_rates, list) and spike_rates.dtype != 'object':
            spike_rates = np.ascontiguousarray(spike_rates, dtype=float)

    else:
        spike_rates = None  # Other ranks initialize as None

    # Broadcast spike_rates to all processes
    try:
        spike_rates = Bcast_from_root(spike_rates, comm)
    except KeyError:
        spike_rates = comm.bcast(spike_rates)


    globals()['X'] = spike_rates
    globals()['data_file'] = data_file
    if broadcast_behavior:
        behavior = dat['behavior'] if (comm is None or comm.rank == 0) else None
        behavior = comm.bcast(behavior) if comm else behavior
        globals()['Y'] = behavior

def dimreduc_(dim_vals, n_folds, comm, method, method_args, results_file, resume=False, stratified_KFold=False):

    results_folder = results_file.split('.')[0]
    if comm is None or comm.rank == 0:
        
        os.makedirs(results_folder, exist_ok=True)
        X, Y = globals()['X'], globals()['Y']
        
        # Perform cross-validation splits
        train_test_idxs = list(KFold(n_folds, shuffle=False).split(X)) if n_folds > 1 else [(list(range(X.shape[0])), [])]

        # Create data task list
        data_tasks = [(idx,) + train_test_split for idx, train_test_split in enumerate(train_test_idxs)]   
        tasks = [task + (method, method_args, results_folder) for task in itertools.product(data_tasks, dim_vals)]
        if resume: tasks = prune_tasks(tasks, results_folder, 'dimreduc')

    else:
        tasks = None


    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map
    pool = MPIPool(comm) if comm else SerialPool()
    if comm is not None: tasks = comm.bcast(tasks)
    print('%d Tasks Remaining' % len(tasks))
    
    worker = PoolWorker()
    if len(tasks) > 0: pool.map(worker.dimreduc, tasks)
    pool.close()

    consolidate(results_folder, results_file, comm)

def decoding_(dimreduc_file, decoder, data_path, comm, results_file, resume=False, loader_args=None):

    # Create folder for processes to write in
    results_folder = results_file.split('.')[0]
    if comm is None or comm.rank == 0:
        os.makedirs(results_folder, exist_ok=True)

    # Look for an arg file in the same folder as the dimreduc_file
    dimreduc_path = '/'.join(dimreduc_file.split('/')[:-1])
    dimreduc_fileno = int(dimreduc_file.split('_')[-1].split('.dat')[0])
    argfile_path = '%s/arg%d.dat' % (dimreduc_path, dimreduc_fileno)

    # Dimreduc args provide loader information
    with open(argfile_path, 'rb') as f:
        args = pickle.load(f) 

    data_file_name = args['data_file'].split('/')[-1]
    data_file_path = '%s/%s' % (data_path, data_file_name)

    # Don't do this one
    if data_file_name == 'trialtype0.dat': return


    # Load Data w/ loader args
    load_data(args['loader'], args['data_file'], loader_args if loader_args is not None else args['loader_args'], comm, broadcast_behavior=True)
    
    
    if comm is None or comm.rank == 0:
        with open(dimreduc_file, 'rb') as f:
            dimreduc_results = pickle.load(f)

        dim_vals = args['task_args']['dim_vals']
        n_folds = args['task_args']['n_folds']
        fold_idxs = np.arange(n_folds)

        tasks = list(itertools.product(dim_vals, fold_idxs))
        dim_fold_tuples = [(result['dim'], result['fold_idx']) for result in dimreduc_results]

        # Match tasks with corresponding dimreduc_results
        for i, task in enumerate(tasks):
            dimreduc_idx = dim_fold_tuples.index((task[0], task[1]))
            tasks[i] += (dimreduc_results[dimreduc_idx], decoder, results_folder)

        if resume: tasks = prune_tasks(tasks, results_folder, 'decoding')
        if comm is not None and comm.rank == 0:
            with open('tasks.pkl', 'wb') as f:
                pickle.dump(tasks, f)
    else:
        tasks = None


    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map
    pool = MPIPool(comm) if comm else SerialPool()
    if comm is not None: tasks = comm.bcast(tasks)
    print('%d Tasks Remaining' % len(tasks))
    
    worker = PoolWorker()
    if len(tasks) > 0: pool.map(worker.decoding, tasks)
    pool.close()

    consolidate(results_folder, results_file, comm)

def main(cmd_args, args):

    # MPI split
    comm = MPI.COMM_WORLD if not cmd_args.serial else None
    ncomms = cmd_args.ncomms if not cmd_args.serial else None          


    if cmd_args.analysis_type == 'dimreduc':

        load_data(args['loader'], args['data_file'], args['loader_args'], comm, broadcast_behavior=True)        
        dimreduc_(dim_vals = args['task_args']['dim_vals'],
                  n_folds = args['task_args']['n_folds'], 
                  comm=comm, 
                  method = args['task_args']['dimreduc_method'],
                  method_args = args['task_args']['dimreduc_args'],
                  results_file = args['results_file'],
                  resume=cmd_args.resume)

    elif cmd_args.analysis_type == 'decoding':
        
        decoding_loader_args = args['loader_args'] if args['loader_args'] else None
        decoding_(dimreduc_file=args['task_args']['dimreduc_file'], 
                  decoder=args['task_args']['decoder'],
                  data_path = args['data_path'], comm=comm, 
                  results_file=args['results_file'],
                  resume=cmd_args.resume,
                  loader_args=decoding_loader_args)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg_file')
    parser.add_argument('--analysis_type', dest='analysis_type')
    parser.add_argument('--serial', dest='serial', action='store_true')
    parser.add_argument('--ncomms', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    cmd_args = parser.parse_args()

    with open(cmd_args.arg_file, 'rb') as f:
        args = pickle.load(f)

    if isinstance(args, dict):
        main(cmd_args, args)
        
    else:
        for arg in args:
            try:
                main(cmd_args, arg)
            except:
                continue
