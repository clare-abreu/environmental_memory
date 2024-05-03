#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 22:43:28 2023

@author: clare
"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import timeit
import time

from scipy import stats
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.spatial import distance
from matplotlib import colors
import scipy
import sys
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
# from sklearn.decomposition import PCA
from ast import literal_eval
from Bio import SeqIO
import itertools
from itertools import combinations
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

sns.set_color_codes()
sns.set_style('white')
sns.set_style('ticks')

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%% Import:

non_neutralAB_withT0_fitness = pd.read_csv('../formatted_data/non_neutralAB_fitness_withT0.csv',index_col=0)
BCs = non_neutralAB_withT0_fitness.index.tolist()

with open('skipz_static_data.pkl', 'rb') as f:
    skipz_static_data = pickle.load(f)
    
with open('skipz_static_data_H_exception.pkl', 'rb') as f:
    skipz_static_data_H_exception = pickle.load(f)
    
with open('fluct_comp1_data.pkl', 'rb') as f:
    fluct_comp1_data = pickle.load(f)

with open('fluct_comp2_data_withT0.pkl', 'rb') as f:
    fluct_comp2_data_withT0 = pickle.load(f)

with open('fluct_comp2_data_withoutT0.pkl', 'rb') as f:
    fluct_comp2_data_withoutT0 = pickle.load(f)

with open('fluct_comp2_data_withT0_exceptions.pkl', 'rb') as f:
    fluct_comp2_data_withT0_exceptions = pickle.load(f)

#%% Functions:
    
def bootstrap_fitness(data_dict,conds,BCs,num_trials):
    bootstrap_dict = {}
    
    for cond in conds:
        bootstrap_dict[cond] = {}
        for BC in BCs:
            bootstrap_dict[cond][BC] = []
            fit_list = data_dict[cond][BC]
            
            if len(fit_list)>2:
                for t in range(num_trials):
                    data = np.random.choice(fit_list, size=len(fit_list), replace=True)
                    
                    def neg_log_likelihood(params):
                        mean, std = params
                        likelihood = np.sum(-np.log(std) - ((data - mean)**2) / (2 * std ** 2))
                        return -likelihood
                    
                    # Initial guess for mean and std
                    initial_guess = [0, 0.1]
                    # Add bounds to the parameters (mean can be any real number; std must be > 0)
                    param_bounds = [(-np.inf, np.inf), (1e-6, np.inf)]
                    # Run the optimizer with bounds
                    result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B', bounds=param_bounds)
                    # Extract the estimated parameters
                    estimated_mean, estimated_std = result.x
                    bootstrap_dict[cond][BC].append(estimated_mean)
                    
    return(bootstrap_dict)
    


#%%
num_trials=150
static_conds = ['Glu','Gal','L','N']
static_exc_cond = ['high_H_H']
fluct_conds = ['Gal-N','Glu-Gal','Glu-L','Glu-N','high_H_Gal-H','high_H_H-N','L-N']
fluct_exc_conds = ['Gal-L','high_H_Glu-H','high_H_L-H']

bootstrap_skipz_static_data = bootstrap_fitness(skipz_static_data,static_conds,BCs,num_trials)
with open('bootstrap_data/bootstrap_skipz_static_data_Sept5.pkl', 'wb') as f:
    pickle.dump(bootstrap_skipz_static_data, f)

bootstrap_skipz_static_data_H_exception = bootstrap_fitness(skipz_static_data_H_exception,static_exc_cond,BCs,num_trials)
with open('bootstrap_data/bootstrap_skipz_static_data_H_exception_Sept5.pkl', 'wb') as f:
    pickle.dump(bootstrap_skipz_static_data_H_exception, f)
    
bootstrap_fluct_comp1_data = bootstrap_fitness(fluct_comp1_data,fluct_conds+fluct_exc_conds,BCs,num_trials)
with open('bootstrap_data/bootstrap_fluct_comp1_data_Sept5.pkl', 'wb') as f:
    pickle.dump(bootstrap_fluct_comp1_data, f)
    
bootstrap_fluct_comp2_data_withT0 = bootstrap_fitness(fluct_comp2_data_withT0,fluct_conds,BCs,num_trials)
with open('bootstrap_data/bootstrap_fluct_comp2_data_withT0_Sept5.pkl', 'wb') as f:
    pickle.dump(bootstrap_fluct_comp2_data_withT0, f)
    
bootstrap_fluct_comp2_data_withT0_exceptions = bootstrap_fitness(fluct_comp2_data_withT0_exceptions,fluct_exc_conds,BCs,num_trials)
with open('bootstrap_data/bootstrap_fluct_comp2_data_withT0_exceptions_Sept5.pkl', 'wb') as f:
    pickle.dump(bootstrap_fluct_comp2_data_withT0_exceptions, f)
    
bootstrap_fluct_comp2_data_withoutT0 = bootstrap_fitness(fluct_comp2_data_withoutT0,fluct_conds,BCs,num_trials)
with open('bootstrap_data/bootstrap_fluct_comp2_data_withoutT0_Sept5.pkl', 'wb') as f:
    pickle.dump(bootstrap_fluct_comp2_data_withoutT0, f)

#%% 3rd round for H2O2
num_trials=250
static_conds = ['Glu','Gal','L','N']
static_exc_cond = ['high_H_H']
fluct_conds = ['Gal-N','Glu-Gal','Glu-L','Glu-N','high_H_Gal-H','high_H_H-N','L-N']
fluct_exc_conds = ['Gal-L','high_H_Glu-H','high_H_L-H']

bootstrap_skipz_static_data_H_exception = bootstrap_fitness(skipz_static_data_H_exception,static_exc_cond,BCs,num_trials)
with open('bootstrap_data/bootstrap_skipz_static_data_H_exception_Sept6.pkl', 'wb') as f:
    pickle.dump(bootstrap_skipz_static_data_H_exception, f)


