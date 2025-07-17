# ------------------------------------------------------------------ Public Packages
import os
import argparse
import pickle
import glob
import itertools
import re
import numpy as np

from mpi_utils.ndarray import Bcast_from_root
from sklearn.model_selection import KFold

# ------------------------------------------------------------------ Custom Built Packages
from dimreduc_wrappers import PCA_wrapper, NoDimreduc, RandomDimreduc
from loaders import (load_sabes, load_peanut, load_sabes_trialized, load_AllenVC)
from decoders import (lr_decoder, rrlr_decoder, logreg, lr_residual_decoder, svm_decoder, psid_decoder)
try:
    from FCCA.fcca import FCCA as LQGCA
except:
    from FCCA_private.FCCA.fcca import LQGComponentsAnalysis as LQGCA


# ------------------------------------------------------------------ Reference Dictionaries
LOADER_DICT = {'sabes': load_sabes, 'peanut': load_peanut, 'sabes_trialized': load_sabes_trialized, 'AllenVC':load_AllenVC}
DECODER_DICT = {'lr': lr_decoder, 'lr_residual': lr_residual_decoder, 'svm':svm_decoder, 'psid':psid_decoder, 'rrlr': rrlr_decoder, 'logreg':logreg}
DIMREDUC_DICT = {'PCA': PCA_wrapper, 'LQGCA': LQGCA, 'None':NoDimreduc, 'Random': RandomDimreduc}


# ------------------------------------------------------------------ Parallelization Functions
def comm_split(comm, ncomms):

    if comm is not None:    
        subcomm = None
        split_ranks = None
    else:
        split_ranks = None

    return split_ranks

def init_comm(comm, split_ranks):

    ncomms = len(split_ranks)
    color = [i for i in np.arange(ncomms) if comm.rank in split_ranks[i]][0]
    return subcomm

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
            _, _, train_test_tuple, dim, _, _, _ = task
            fold_idx, _, _ = train_test_tuple

            if (dim, fold_idx) not in param_tuples:
                to_do.append(task)

        elif task_format == 'decoding':
            _, _, dim, fold_idx, _, _, _ = task

            if (dim, fold_idx) not in param_tuples:
                to_do.append(task)


    return to_do

def consolidate(results_folder, results_file, comm):
    # Consolidate files into a single data file
    if comm is not None:
        if comm.rank == 0:
            data_files = glob.glob('%s/*.dat' % results_folder)
            results_dict_list = []
            for data_file in data_files:
                with open(data_file, 'rb') as f:
                    try:
                        results_dict = pickle.load(f)
                    except:
                        # Delete the data file since something went wrong
                        os.remove(data_file)
                        return
                    results_dict_list.append(results_dict)

            with open(results_file, 'wb') as f:
                f.write(pickle.dumps(results_dict_list))
    else:
        data_files = glob.glob('%s/*.dat' % results_folder)
        results_dict_list = []
        for data_file in data_files:
            with open(data_file, 'rb') as f:
                try:
                    results_dict = pickle.load(f)
                except:
                    # Delete the data file since something went wrong
                    os.remove(data_file)
                    return

                results_dict_list.append(results_dict)
        
        with open(results_file, 'wb') as f:    
            f.write(pickle.dumps(results_dict_list))

class PoolWorker():

    # Initialize the worker with the data so it does not have to be broadcast by pool.map
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

  
    def dimreduc(self, task_tuple):
        
        task_tuple, comm = task_tuple if len(task_tuple) == 2 else (task_tuple, None)            

        #train_test_tuple, dim, method, method_args, results_folder = task_tuple
        task_idx, total_tasks, train_test_tuple, dim, method, method_args, results_folder = task_tuple
        fold_idx, train_idxs, test_idxs = train_test_tuple
        print(f"[Task {task_idx+1}/{total_tasks}] Method: {method}, Dim: {dim}, Fold: {fold_idx}")

        
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
        task_idx, total_tasks, dim_val, fold_idx, dimreduc_results, decoder, results_folder = task_tuple

        print(f"[Task {task_idx+1}/{total_tasks}] Decoder: {decoder['method']}, Dim: {dim_val}, Fold: {fold_idx}")
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
                
                results = DECODER_DICT[decoder['method']](Xtest_, Xtrain_, Ytest_, Ytrain_, **decoder['args'])
                results_dict = {**dimreduc_results, **results}
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
            if np.ndim(coef_) == 3: coef_ = np.reshape(np.squeeze(coef_), (coef_.shape[0],1))

            Xtrain = Xtrain @ coef_ if np.ndim(Xtrain) == 2 else [xx @ coef_ for xx in Xtrain]
            Xtest = Xtest @ coef_ if np.ndim(Xtest) == 2 else [xx @ coef_ for xx in Xtest]
            Ytrain, Ytest = list(Ytrain), list(Ytest)  # Convert to list if needed

            results = DECODER_DICT[decoder['method']](Xtest, Xtrain, Ytest, Ytrain, **decoder['args'])
            results_dict = {**dimreduc_results, **results}
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

    if comm is not None:
        try:
            # Broadcast spike_rates to all processes
            spike_rates = Bcast_from_root(spike_rates, comm)
        except KeyError:
            spike_rates = comm.bcast(spike_rates)


    globals()['X'] = spike_rates
    globals()['data_file'] = data_file
    if broadcast_behavior:
        behavior = dat['behavior'] if (comm is None or comm.rank == 0) else None
        behavior = comm.bcast(behavior) if comm else behavior
        globals()['Y'] = behavior

def dimreduc_(dim_vals, n_folds, comm, method, method_args, results_file, resume=False):

    results_folder = results_file.split('.')[0]
    if comm is None or comm.rank == 0:
        
        os.makedirs(results_folder, exist_ok=True)
        X, Y = globals()['X'], globals()['Y']
        
        # Perform cross-validation splits
        train_test_idxs = list(KFold(n_folds, shuffle=False).split(X)) if n_folds > 1 else [(list(range(X.shape[0])), [])]

        # Create data task list
        data_tasks = [(idx,) + train_test_split for idx, train_test_split in enumerate(train_test_idxs)]   
        task_list = list(itertools.product(data_tasks, dim_vals))
        total_tasks = len(task_list)
        tasks = [(i, total_tasks, *task, method, method_args, results_folder) for i, task in enumerate(task_list)]
        
        if resume: tasks = prune_tasks(tasks, results_folder, 'dimreduc')

    else:
        tasks = None

    if comm is not None: tasks = comm.bcast(tasks)
    
    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map
    #pool = MPIPool(comm) if comm else SerialPool()
   #if comm is not None: tasks = comm.bcast(tasks)
    
    if comm is None:
        worker = PoolWorker()
        for task in tasks:
            worker.dimreduc(task)
    else:
        from schwimmbad import MPIPool
        pool = MPIPool(comm)
        if len(tasks) > 0:
            pool.map(PoolWorker().dimreduc, tasks)
        pool.close()    
    

    consolidate(results_folder, results_file, comm)

def decoding_(dimreduc_file, decoder, data_path,
              comm, results_file, 
              resume=False, loader_args=None):

    if comm is not None:
        # Create folder for processes to write in
        results_folder = results_file.split('.')[0]
        if comm.rank == 0:
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
    else: 
        results_folder = results_file.split('.')[0]        
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    # Look for an arg file in the same folder as the dimreduc_file
    dimreduc_path = '/'.join(dimreduc_file.split('/')[:-1])
    dimreduc_fileno = int(dimreduc_file.split('_')[-1].split('.dat')[0])
    argfile_path = '%s/arg%d.dat' % (dimreduc_path, dimreduc_fileno)


    # Dimreduc args provide loader information
    with open(argfile_path, 'rb') as f:
        args = pickle.load(f) 


    if loader_args is not None:
        load_data(args['loader'], args['data_file'], loader_args, comm, broadcast_behavior=True)
    else:
        load_data(args['loader'], args['data_file'], args['loader_args'], comm, broadcast_behavior=True)
    
    if comm is None:
        with open(dimreduc_file, 'rb') as f:
            dimreduc_results = pickle.load(f)
        dim_vals = args['task_args']['dim_vals']
        n_folds = args['task_args']['n_folds']
        fold_idxs = np.arange(n_folds)

        # Assemble task arguments
        tasks = list(itertools.product(dim_vals, fold_idxs))

        dim_fold_tuples = [(result['dim'], result['fold_idx']) for result in dimreduc_results]

        task_list = []
        for i, (dim_val, fold_idx) in enumerate(itertools.product(dim_vals, fold_idxs)):
            dimreduc_idx = dim_fold_tuples.index((dim_val, fold_idx))
            task_list.append((i, len(dim_vals)*len(fold_idxs), dim_val, fold_idx, dimreduc_results[dimreduc_idx], decoder, results_folder))
        tasks = task_list

        if resume:
            tasks = prune_tasks(tasks, results_folder, 'decoding')
    else:
        if comm.rank == 0:
            with open(dimreduc_file, 'rb') as f:
                dimreduc_results = pickle.load(f)

            # Pass in for manual override for use in cleanup
            dim_vals = args['task_args']['dim_vals']
            n_folds = args['task_args']['n_folds']
            fold_idxs = np.arange(n_folds)
                
            tasks = list(itertools.product(dim_vals, fold_idxs))
            fold_idxs = np.arange(n_folds)
            dim_fold_tuples = [(result['dim'], result['fold_idx']) for result in dimreduc_results]


            task_list = []
            for i, (dim_val, fold_idx) in enumerate(itertools.product(dim_vals, fold_idxs)):
                dimreduc_idx = dim_fold_tuples.index((dim_val, fold_idx))
                task_list.append((i, len(dim_vals)*len(fold_idxs), dim_val, fold_idx, dimreduc_results[dimreduc_idx], decoder, results_folder))
            tasks = task_list

            if resume:
                tasks = prune_tasks(tasks, results_folder, 'decoding')
            with open('tasks.pkl', 'wb') as f:
                f.write(pickle.dumps(tasks))
                
        else:
            tasks = None

    # Initialize Pool worker with data
    worker = PoolWorker()

    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map

    if comm is not None and comm.Get_size() > 1:
        tasks = comm.bcast(tasks)
        from schwimmbad import MPIPool
        pool = MPIPool(comm)
    else:
        from schwimmbad import SerialPool
        pool = SerialPool()

    if len(tasks) > 0:
        pool.map(worker.decoding, tasks)

    pool.close()

    consolidate(results_folder, results_file, comm)

def main(cmd_args, args):

    # MPI split
    #comm = MPI.COMM_WORLD if not cmd_args.serial else None
    #ncomms = cmd_args.ncomms if not cmd_args.serial else None          
    if cmd_args.serial:
        comm = None
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() == 1:
            comm = None

    ncomms = cmd_args.ncomms if comm is not None else None


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

    print("Finished script")
    
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
