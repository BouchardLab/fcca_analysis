import matplotlib.pyplot as plt
import scipy
import numpy as np
import pdb
import glob
import pickle
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from region_select import *
from config import PATH_DICT

import sys
sys.path.append(PATH_DICT['repo'])
from nn_multinomial import *
from decoders import logreg_preprocess

from mpi4py import MPI

class NucNormLogReg(LogisticRegression):
    def __init__(self, lambda_reg=0, max_iter=1000, s=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.s = s

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        alpha, B, _, _, _, _, _, _ = accelerated_proximal_gradient_descent(X, y, 
                                                                           lambda_reg=self.lambda_reg, 
                                                                           max_iter=self.max_iter, 
                                                    
                                                                           s=self.s, verbose=False)
        self.coef_ = B.T
        self.intercept_ = alpha.squeeze()
        return self

def compare_to_sklearn(dat):

    lr_scores = []
    nn_scores = []
    reduc_scores_all = []

    for train_idxs, test_idxs in \
        KFold(n_splits=5).split(np.arange(dat['spike_rates'].shape[0])):
        
        Xtrain = dat['spike_rates'][train_idxs]
        Xtest = dat['spike_rates'][test_idxs]
        Ytrain = dat['behavior'][train_idxs]
        Ytest = dat['behavior'][test_idxs]

        Xtrain, Ytrain = logreg_preprocess(Xtrain, Ytrain)
        Xtest, Ytest = logreg_preprocess(Xtest, Ytest)

        lr = LogisticRegression()
        lr.fit(Xtrain, Ytrain)
        lr_scores.append(lr.score(Xtest, Ytest))

        # Perform an SVD on the sklearn solution and see how well this does
        u, s, vh = np.linalg.svd(lr.coef_)
        dims = np.arange(1, s.size, 2)
        reduc_scores = []
        for k in tqdm(dims):
            coef_reduc = u[:, :k] @ np.diag(s[0:k]) @ vh[:k]
            lr.coef_ = coef_reduc
            reduc_scores.append(lr.score(Xtest, Ytest))

        reduc_scores_all.append(reduc_scores)

    # Average over folds
    reduc_scores_all = np.array(reduc_scores_all)
    return reduc_scores_all, lr_scores, dims

def inspect_rank(dat):
    # Over a range of lambda_regs, inspect the rank of the result B matrix
    lambda_regs = np.logspace(-3, 1, 50)

    train_idxs, test_idxs = next(KFold(n_splits=5).split(np.arange(dat['spike_rates'].shape[0])))
    Xtrain = dat['spike_rates'][train_idxs]
    Ytrain = dat['behavior'][train_idxs]

    Xtest = dat['spike_rates'][test_idxs]
    Ytest = dat['behavior'][test_idxs]

    Xtrain, Ytrain = logreg_preprocess(Xtrain, Ytrain)
    Xtest, Ytest = logreg_preprocess(Xtest, Ytest)

    ranks = []
    for lambda_reg in tqdm(lambda_regs):
        lr = NucNormLogReg(lambda_reg=lambda_reg, max_iter=3000, s=0.01)
        lr.fit(Xtrain, Ytrain)
        s = np.linalg.svd(lr.coef_, compute_uv=False)
        ranks.append(s)
    return lambda_regs, ranks

def fit(dat):
    lambda_regs = np.logspace(-3, 1, 50)

    for fold_idx, (train_idxs, test_idxs) in \
        enumerate(KFold(n_splits=5).split(np.arange(dat['spike_rates'].shape[0]))):

        Xtrain = dat['spike_rates'][train_idxs]
        Ytrain = dat['behavior'][train_idxs]

        Xtest = dat['spike_rates'][test_idxs]
        Ytest = dat['behavior'][test_idxs]

        Xtrain, Ytrain = logreg_preprocess(Xtrain, Ytrain)
        Xtest, Ytest = logreg_preprocess(Xtest, Ytest)

        slist = []
        scores = []

        # Execute in parallel

        for lambda_reg in tqdm(lambda_regs):
            lr1 = NucNormLogReg(lambda_reg=lambda_reg, max_iter=3000, s=0.01)
            lr1.fit(Xtrain, Ytrain)
            scores1 = lr1.score(Xtest, Ytest)
            
            lr2 = NucNormLogReg(lambda_reg=lambda_reg, max_iter=3000, s=0.01)
            lr2.fit(Xtrain, Ytrain)
            scores2 = lr2.score(Xtest, Ytest)
            
            # Do we get the same solution with multiple runs?
            pdb.set_trace()

            s = np.linalg.svd(lr.coef_, compute_uv=False)
            slist.append(s)
            scores.append(lr.score(Xtest, Ytest))

        # Save the results
        with open(f'/home/ankit_kumar/Data/FCCA_revisions/VISP_nn_logreg/{session}_{fold_idx}.pkl', 'wb') as f:
            pickle.dump((lambda_regs, slist, scores), f)

# Consolidate results of fit into a dataframe
def consolidate_results(dimreduc_decoding_df, path):

    # What do we truncate singular values at?
    svd_thresh = 1e-8

    # Get fit results
    fit_results = glob.glob(f'{path}/*.pkl')

    # Get the session names
    sessions = np.unique([os.path.basename(f).split('.')[0] for f in fit_results])
    # Copy over needed columns
    loader_args = dimreduc_decoding_df.iloc[0]['loader_args']

    # Dim key will be rank
    dim_key = 'rank'
    result_list = []
    for fit_result in tqdm(fit_results):
        session = os.path.basename(fit_result).split('.')[0]
        fold_idx = int(os.path.basename(fit_result).split('nwb_')[1].split('.')[0])
        with open(fit_result, 'rb') as f:
            lambda_regs, slist, scores = pickle.load(f)
            ranks = []
            for s in slist:
                ranks.append(s[s > svd_thresh].size)
            
            unique_ranks, unique_rank_idxs = np.unique(ranks, return_index=True)
            for i, rank in zip(unique_rank_idxs, unique_ranks):
                if rank == 0:
                    continue
                result = {
                    'loader_args': loader_args,
                    'data_file': f'{session}.nwb',
                    'fold_idx': fold_idx,
                    'rank': rank,
                    'lambda_reg': lambda_regs[i],
                    'accuracy': scores[i],
                }
                result_list.append(result)

    # Convert to dataframe and save
    result_df = pd.DataFrame(result_list)
    with open(PATH_DICT['df'] + '/visp_supervised_df.pkl', 'wb') as f:
        pickle.dump(result_df, f)
    return result_df

def debug_results(dimreduc_decoding_df, path):
    # Verify that there is a monotonic relationship between rank and accuracy
    # --> they are not.
    fit_results = glob.glob(f'{path}/*.pkl')

    # Get the session names
    sessions = np.unique([os.path.basename(f).split('.')[0] for f in fit_results])
    # Copy over needed columns
    loader_args = dimreduc_decoding_df.iloc[0]['loader_args']

    # Dim key will be rank
    dim_key = 'rank'
    result_list = []
    for fit_result in tqdm(fit_results):
        session = os.path.basename(fit_result).split('.')[0]
        fold_idx = int(os.path.basename(fit_result).split('nwb_')[1].split('.')[0])
        with open(fit_result, 'rb') as f:
            lambda_regs, slist, scores = pickle.load(f)
            pdb.set_trace()

    # Refit with balanced classes



# First, check the performance of our approach without nuclear norm regualarization
# against sklearn's logistic regression
if __name__ == '__main__':
    region = 'VISp'
    save_path = '/home/ankit_kumar/Data/FCCA_revisions/VISP_logreg_svd/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ##### Fitting #####
    if False:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            df, session_key = load_decoding_df(region,  **loader_kwargs[region])
            sessions = df[session_key].unique()
            data_path = get_data_path(region)
            loader_args = df.iloc[0]['loader_args']
        else:
            sessions = None
            data_path = None
            loader_args = None

        sessions = comm.bcast(sessions, root=0)
        data_path = comm.bcast(data_path, root=0)
        loader_args = comm.bcast(loader_args, root=0)
        sessions = np.array_split(sessions, size)[rank]
        for session in sessions:                           
            dat = load_data(data_path, region, session, loader_args=loader_args)
            reduc_scores, lr_scores, dims = compare_to_sklearn(dat)

            # Save the results
            with open(f'{save_path}/{session}_reduc_scores.pkl', 'wb') as f:
                pickle.dump((reduc_scores, lr_scores, dims), f)

    if True:
        df, session_key = load_decoding_df(region,  **loader_kwargs[region])
        sessions = df[session_key].unique()
        data_path = get_data_path(region)
        loader_args = df.iloc[0]['loader_args']

        # Load the results from above
        fls = glob.glob(f'{save_path}/*.pkl')
        reduc_scores_all = []
        lr_scores_all = []
        dims_all = []
        sessions = []
        for f in fls:
            sessions.append(os.path.basename(f).split('_reduc_scores')[0])
            with open(f, 'rb') as f:
                reduc_scores, lr_scores, dims = pickle.load(f)
                reduc_scores_all.append(reduc_scores)
                lr_scores_all.append(lr_scores)
                dims_all.append(dims)

        min_size = np.min([r.shape[1] for r in reduc_scores_all])
        reduc_scores_all = [r[:, :min_size] for r in reduc_scores_all]
        dims_all = [r[:min_size] for r in dims_all]
        reduc_scores_all = np.array(reduc_scores_all)
        lr_scores_all = np.array(lr_scores_all)
        dims_all = np.array(dims_all)

        # Save away as a list of dictionaries for downstream plotting
        result_list = []
        for i, session in enumerate(sessions):
            for j, fold_idx in enumerate(range(reduc_scores_all.shape[1])):
                for k, dim in enumerate(dims_all[i]):
                    result_list.append({
                        # Match session_key to other visp files
                        'data_file': session,
                        'fold_idx': fold_idx,
                        'dim': dim,
                        'acc': reduc_scores_all[i, j, k],
                        'loader_args': loader_args
                })

        with open(PATH_DICT['df'] + '/visp_logreg_svd.pkl', 'wb') as f:
            pickle.dump(result_list, f)

        lr_scores_all = lr_scores_all[..., np.newaxis]
        lr_scores_all = np.tile(lr_scores_all, (1, 1, reduc_scores_all.shape[-1]))

        # Calculate mean and standard error across sessions and folds
        mean_reduc_scores = np.mean(reduc_scores_all, axis=(0, 1))
        se_reduc_scores = np.std(reduc_scores_all, axis=(0, 1)) / np.sqrt(reduc_scores_all.shape[0] * reduc_scores_all.shape[1])
        
        # For lr_scores, we need to calculate the mean and SE
        mean_lr_scores = np.mean(lr_scores_all)
        se_lr_scores = np.std(lr_scores_all) / np.sqrt(np.prod(lr_scores_all.shape))
        
        # Use the first dims array since they should all be the same
        dims_to_plot = dims_all[0]
        pdb.set_trace()
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot reduced rank scores with error bars
        plt.errorbar(dims_to_plot, mean_reduc_scores, yerr=se_reduc_scores, 
                    marker='o', linestyle='-', label='Reduced Rank')
        
        # Plot full rank (lr) scores as a horizontal line with error band
        plt.axhline(y=mean_lr_scores, color='r', linestyle='--', label='Full Rank')
        plt.fill_between(dims_to_plot, 
                        mean_lr_scores - se_lr_scores, 
                        mean_lr_scores + se_lr_scores, 
                        color='r', alpha=0.2)
        
        plt.xlabel('Rank')
        plt.ylabel('Decoding Accuracy')
        plt.title('Decoding Performance vs. Rank')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(f'./rank_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()



    #### Consolidation and debugging ####
    # df, session_key = load_decoding_df(region,  **loader_kwargs[region])
    # # consolidate_results(df, '/home/ankit_kumar/Data/FCCA_revisions/VISP_nn_logreg/')
    # debug_results(df, '/home/ankit_kumar/Data/FCCA_revisions/VISP_nn_logreg/')

