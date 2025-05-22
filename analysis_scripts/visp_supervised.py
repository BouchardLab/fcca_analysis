from region_select import *
from config import *
import sys, os
from tqdm import tqdm
sys.path.append(PATH_DICT['repo'])
from loaders import load_AllenVC
from sklearn.model_selection import KFold
from decoders import logreg_preprocess
from sklearn.linear_model import LogisticRegression
import pickle
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter 
# from rpy2.robjects import numpy2ri  

# numpy2ri.activate()

# # r('library(npmr)')
# npmr = importr('npmr')

data_path = get_data_path('VISp')


# Taken from AllenVC_dimreduc_args.py
session_IDs = [732592105, 754312389, 798911424, 791319847, 754829445, 760693773, 757216464, 797828357, 762120172, 757970808, 799864342, 762602078, 755434585, 763673393, 760345702, 750332458, 715093703, 759883607, 719161530, 750749662, 756029989]

data_files = [os.path.join(data_path, 
                           f"session_{session_ID}", 
                           f"session_{session_ID}.nwb") 
                           for session_ID in session_IDs]


loader_args =  {'region': 'VISp', 'bin_width':15, 
                             'preTrialWindowMS':0, 
                              'postTrialWindowMS':0, 
                              'boxcox':0.5}

# Can't necessarily use this
dimvals = np.arange(1, 48)

scores = np.zeros((len(session_IDs), 5))

for i, session in tqdm(enumerate(session_IDs)):
    dat = load_AllenVC(data_files[i], **loader_args)
    print(dat.keys())



    X = dat['spike_rates']
    # Natural image labels
    y = dat['behavior']

    # Cross-validate
    for j, (train_idx, test_idx) in enumerate(KFold(n_splits=5).split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train decoder
        x_out, y_out = logreg_preprocess(X_train, y_train)

        logreg = LogisticRegression()   
        logreg.fit(x_out, y_out)
        
        xtest, ytest = logreg_preprocess(X_test, y_test)
        scores[i, j] = logreg.score(xtest, ytest)

        # s_default = 0.1/np.max(x_out)
        # npmr.npmr(x_out, y_out[:, np.newaxis], 
        # s_default = 0.1/np.max(x_out)
        # npmr.npmr(x_out, y_out[:, np.newaxis], 
        #           s=s_default/10, eps=1e-6)

# Save scores
with open('visp_supervised_scores.pkl', 'wb') as f:
    pickle.dump(scores, f)