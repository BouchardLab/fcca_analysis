import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from tqdm import tqdm
import sys, os

from region_select import *
from config import PATH_DICT

from Fig6 import collate_fig6justification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append(PATH_DICT['repo'])

def plot2D(regions, independent_var, response_fcca, response_pca, 
           rand_control, xlabel, ylabel, dimset, include_rand_offset=True):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Create increasingly redder shades from light pink to dark red
    region_colors_fcca = ['#ffcccb', '#ff9999', '#ff6666', '#ff3333', '#ff0000', '#cc0000', '#990000']
    # Create increasingly darker shades of grey/black
    region_colors_pca = ['#cccccc', '#999999', '#666666', '#333333', '#000000']
    # Diff colors - match bamboozle.py
    region_names = ['M1', 'S1', 'HPC', 'VISp']
    region_markers = ['o', 's', 'o', 'o']
    region_colors_diff = ['purple', 'purple', 'g', '#bf842c']

    for i in range(len(regions)):
        x_ = independent_var[i]

        if include_rand_offset:
            y_ = (response_fcca[i] - rand_control[i]) - (response_pca[i] - rand_control[i])
        else:
            y_ = response_fcca[i] - response_pca[i]

        ax.scatter(x_, y_, label=region_names[i], color=region_colors_diff[i],
                     marker=region_markers[i], alpha=0.5, edgecolors=region_colors_diff[i])

    ax.legend(fontsize=12, loc='lower right')

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(r'$\Delta$' + f' {ylabel}', fontsize=14)

    ax.set_title(f'Random offset: {include_rand_offset}')
    fig.tight_layout()
    fname = f'{PATH_DICT['figs']}/fig6_justification/{xlabel}vs{ylabel}_randoffset_{include_rand_offset}_dimset{dimset}.pdf'
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot3D(regions, independent_var1, independent_var2, response_fcca, response_pca, 
           rand_control, xlabel, ylabel, zlabel, dimset, include_rand_offset=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create increasingly redder shades from light pink to dark red
    region_colors_fcca = ['#ffcccb', '#ff9999', '#ff6666', '#ff3333', '#ff0000', '#cc0000', '#990000']
    # Create increasingly darker shades of grey/black
    region_colors_pca = ['#cccccc', '#999999', '#666666', '#333333', '#000000']
    # Diff colors - match bamboozle.py
    region_names = ['M1', 'S1', 'HPC', 'VISp']
    region_markers = ['o', 's', 'o', 'o']
    region_colors_diff = ['purple', 'purple', 'g', '#bf842c']

    for i in range(len(regions)):
        x_ = independent_var1[i]
        y_ = independent_var2[i]

        if include_rand_offset:
            z_ = (response_fcca[i] - rand_control[i]) - (response_pca[i] - rand_control[i])
        else:
            z_ = response_fcca[i] - response_pca[i]

        ax.scatter(x_, y_, z_, label=region_names[i], color=region_colors_diff[i],
                   marker=region_markers[i], alpha=0.5, edgecolors=region_colors_diff[i])

    ax.legend(fontsize=12, loc='best')

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_zlabel(r'$\Delta$' + f' {zlabel}', fontsize=14)

    ax.set_title(f'Random offset: {include_rand_offset}')
    # change orientation
    ax.view_init(elev=45, azim=-210)

    fig.tight_layout()
    fname = f'{PATH_DICT["figs"]}/fig6_justification/{xlabel}_{ylabel}vs{zlabel}_randoffset_{include_rand_offset}_dimset{dimset}.pdf'
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
    # Also pickle the figure so we can restore it with interactivity
    with open(fname.replace('.pdf', '.pkl'), 'wb') as f:
        pickle.dump(fig, f)
    plt.close(fig)

if __name__ == '__main__':
    if not os.path.exists(PATH_DICT['figs'] + '/fig6_justification'):
        os.makedirs(PATH_DICT['figs'] + '/fig6_justification')
    regions = ['S1_psid', 'HPC_peanut', 'VISp']
    dim_set = 0

    ss_angles_all, rot_dyn_results_dicts, dims, dim_delta_max = \
        collate_fig6justification(regions, dim_set)

    # Plots with and without random offset, for rot strength and dyn range, as well as the 
    # normalized versions of these quantities
    independent_vars = []
    independent_var_names = []
    # peak delta dims
    independent_vars.append(dim_delta_max)
    independent_var_names.append('peak delta dims')
    # FBC/FBCm - average over folds and angles
    independent_vars.append([s[:, :, 2, :].mean(axis=(1, 2)) for s in ss_angles_all])
    independent_var_names.append('FBC_FBCm')
    # FFC/FFCm
    independent_vars.append([s[:, :, 1, :].mean(axis=(1, 2)) for s in ss_angles_all])
    independent_var_names.append('FFC_FFCm')
    # FBC/FBCm - FFC/FFCm
    independent_vars.append([s[:, :, 0, :].mean(axis=(1, 2)) - s[:, :, 1, :].mean(axis=(1, 2)) 
                            for s in ss_angles_all])
    independent_var_names.append('FBC_FBCm_minus_FFC_FFCm')
    # FBC/FBCm/FFC/FFCm
    independent_vars.append([np.divide(s[:, :, 0, :].mean(axis=(1, 2)), s[:, :, 1, :].mean(axis=(1, 2))) 
                            for s in ss_angles_all])
    independent_var_names.append('FBC_FBCm_div_FFC_FFCm')

    # response variables
    response_vars_fcca = []
    response_vars_pca = []
    control_vars = []
    response_var_names = []

    response_vars_fcca.append([rot_dyn_results_dicts[i]['rot_strength_fcca'] for i in range(len(regions))])
    response_vars_pca.append([rot_dyn_results_dicts[i]['rot_strength_pca'] for i in range(len(regions))])
    control_vars.append([rot_dyn_results_dicts[i]['mu_rot_control'] for i in range(len(regions))])
    response_var_names.append('Rot')

    response_vars_fcca.append([rot_dyn_results_dicts[i]['dyn_range_fcca'] for i in range(len(regions))])
    response_vars_pca.append([rot_dyn_results_dicts[i]['dyn_range_pca'] for i in range(len(regions))])
    control_vars.append([rot_dyn_results_dicts[i]['mu_dyn_control'] for i in range(len(regions))])
    response_var_names.append('Dyn')

    # Normalized versions
    response_vars_fcca.append([rot_dyn_results_dicts[i]['rot_strength_maxnorm_fcca'] 
                                for i in range(len(regions))])
    response_vars_pca.append([rot_dyn_results_dicts[i]['rot_strength_maxnorm_pca'] 
                                for i in range(len(regions))])
    control_vars.append([rot_dyn_results_dicts[i]['mu_rot_maxnorm_control'] 
                                for i in range(len(regions))])
    response_var_names.append('Rot_maxnorm')

    response_vars_fcca.append([rot_dyn_results_dicts[i]['dyn_range_maxnorm_fcca'] 
                                for i in range(len(regions))])
    response_vars_pca.append([rot_dyn_results_dicts[i]['dyn_range_maxnorm_pca'] 
                                for i in range(len(regions))])
    control_vars.append([rot_dyn_results_dicts[i]['mu_dyn_maxnorm_control'] 
                                for i in range(len(regions))])
    response_var_names.append('Dyn_maxnorm')

    response_vars_fcca.append([rot_dyn_results_dicts[i]['rot_strength_stdnorm_fcca'] 
                                for i in range(len(regions))])
    response_vars_pca.append([rot_dyn_results_dicts[i]['rot_strength_stdnorm_pca'] 
                                for i in range(len(regions))])
    control_vars.append([rot_dyn_results_dicts[i]['mu_rot_stdnorm_control'] 
                                for i in range(len(regions))])
    response_var_names.append('Rot_stdnorm')

    response_vars_fcca.append([rot_dyn_results_dicts[i]['dyn_range_stdnorm_fcca'] 
                                for i in range(len(regions))])
    response_vars_pca.append([rot_dyn_results_dicts[i]['dyn_range_stdnorm_pca'] 
                                for i in range(len(regions))])
    control_vars.append([rot_dyn_results_dicts[i]['mu_dyn_stdnorm_control'] 
                                for i in range(len(regions))])
    response_var_names.append('Dyn_stdnorm')

    independent_collated = [(iv, ivn) for iv, ivn in zip(independent_vars, independent_var_names)]
    responses_collated = [(rf, rp, cv, n) 
                            for rf, rp, cv, n in zip(response_vars_fcca, 
                                                    response_vars_pca, 
                                                    control_vars, 
                                                    response_var_names)]

    # Take all combinations of indepndent and collated responses
    combs = list(itertools.product(independent_collated, responses_collated))
    for comb in combs:
        plot2D(regions, comb[0][0], comb[1][0], comb[1][1], comb[1][2], 
                comb[0][1], comb[1][3], 
                dim_set, include_rand_offset=True)
        plot2D(regions, comb[0][0], comb[1][0], comb[1][1], comb[1][2], 
                comb[0][1], comb[1][3], 
                dim_set, include_rand_offset=False)        
    
    # Plot 3D versions
    independent_vars_3d = []
    independent_var_names_3d = []

    # Combine peak_delta_dims with the other independent variables
    for i in range(1, len(independent_vars)):
        independent_vars_3d.append([independent_vars[0], independent_vars[i]])
        independent_var_names_3d.append([independent_var_names[0],
                                            independent_var_names[i]])

    independent_collated_3d = [(iv, ivn) 
                                for iv, ivn in zip(independent_vars_3d, 
                                                    independent_var_names_3d)]
    combs_3d = list(itertools.product(independent_collated_3d, responses_collated))
    for comb in combs_3d:
        plot3D(regions, comb[0][0][0], comb[0][0][1], comb[1][0], comb[1][1], 
                comb[1][2], comb[0][1][0], comb[0][1][1], comb[1][3], 
                dim_set, include_rand_offset=True)
        plot3D(regions, comb[0][0][0], comb[0][0][1], comb[1][0], comb[1][1], 
                comb[1][2], comb[0][1][0], comb[0][1][1], comb[1][3], 
                dim_set, include_rand_offset=False)