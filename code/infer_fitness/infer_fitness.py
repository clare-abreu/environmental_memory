#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:01:51 2023

@author: clare
"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pandas as pd
import seaborn as sns
import numpy as np
import pickle

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
    
all_counts = pd.read_csv('../../data/tidy_bc_fitness_counts_freqs.csv',index_col=1)
#BClist = pd.read_csv('../../../data/BClist.csv',index_col=1)

#%% Functions

def calc_mean_fit(all_counts,cond_rep_times):
    # Calculate mean fitness using neutrals at all timepoints in all reps and conds
    # The variable 'timepoints' should not include the first timepoint
    ave_mean_fit = {}
    conds = list(cond_rep_times.keys())
    for cond in conds:
        ave_mean_fit[cond] = {}
        for rep in [1,2,3]:
            timepoints = cond_rep_times[cond][rep-1]
            ave_mean_fit[cond][rep]={}
            these_counts = all_counts[all_counts['condition']==f'{cond}{rep}']
            these_neuts = these_counts[these_counts['neutral']==True]
            neut_BCs = np.unique(these_neuts.index).tolist()
            mean_fit = {}
            for k in timepoints:
                mean_fit[k]=[]
            for neut in neut_BCs:
                for timept in timepoints:
                    total_now = np.sum(these_counts[these_counts['time']==timept]['count'])
                    total_prev = np.sum(these_counts[these_counts['time']==timept-1]['count'])
                    neut_now = these_neuts[these_neuts['time']==timept]
                    neut_prev = these_neuts[these_neuts['time']==timept-1]
                    count_now = neut_now.loc[neut,'count']
                    count_prev = neut_prev.loc[neut,'count']
                    freq_now = count_now/total_now
                    freq_prev = count_prev/total_prev
                    if (count_now > 0) & (count_prev > 0):
                        mean_fit[timept].append(-np.log(freq_now/freq_prev))
                    else:
                        #print('zero')
                        break
            for timept in timepoints:
                ave_mean_fit[cond][rep][timept] = np.mean(mean_fit[timept])
    return ave_mean_fit


def calc_fitness(all_counts,ave_mean_fit,cond_rep_times,rep_sep=True):
    # Calculate mutant fitnesses
    # cond_rep_times is a dictionary where each condition is a dictionary containing 
    # a list of three lists of timepoints to use with each replicate
    # If rep_sep, calculate fitness of each replicate separately

    all_BCs = []
    all_conds = []
    all_reps = []
    all_fits = []
    conds = list(cond_rep_times.keys())
    BCs = np.unique(all_counts.index).tolist()
    for cond in conds:
        for BC in BCs:
            this_fit=[]
            for rep in [1,2,3]:
                if rep_sep:
                    this_fit=[]
                timepoints = cond_rep_times[cond][rep-1]
                these_counts = all_counts[all_counts['condition']==f'{cond}{rep}']
                this_BC = these_counts[these_counts.index==BC]
                for timept in timepoints:
                    total_now = np.sum(these_counts[these_counts['time']==timept]['count'])
                    total_prev = np.sum(these_counts[these_counts['time']==timept-1]['count'])
                    now = this_BC[this_BC['time']==timept]
                    prev = this_BC[this_BC['time']==timept-1]
                    count_now = now.loc[BC,'count']
                    count_prev = prev.loc[BC,'count']
                    freq_now = count_now/total_now
                    freq_prev = count_prev/total_prev
                    if (count_now > 0) & (count_prev > 0):
                        this_fit.append(np.log(freq_now/freq_prev)+ave_mean_fit[cond][rep][timept])
                    else:
                        #print('zero')
                        break
                if rep_sep:
                    if len(this_fit) > 0:
                        all_BCs.append(BC)
                        all_conds.append(cond)
                        all_reps.append(rep)
                        all_fits.append(np.mean(this_fit))
                    else:
                        all_BCs.append(BC)
                        all_conds.append(cond)
                        all_reps.append(rep)
                        all_fits.append(np.nan)
                        
            if rep_sep==False:
                if len(this_fit) > 0:
                    all_BCs.append(BC)
                    all_conds.append(cond)
                    all_fits.append(np.mean(this_fit))
                else:
                    all_BCs.append(BC)
                    all_conds.append(cond)
                    all_fits.append(np.nan)
                
    if rep_sep:        
        data = {
            'BC' : all_BCs,
            'condition' : all_conds,
            'rep' : all_reps,
            'fitness': all_fits
        }
    else:
        data = {
            'BC' : all_BCs,
            'condition' : all_conds,
            'fitness': all_fits
        }
    # creating a Dataframe object 
    df_all_fit = pd.DataFrame(data) 
    df_all_fit.set_index('BC',inplace=True)
    
    return df_all_fit

def save_fitness(all_counts,ave_mean_fit,cond_rep_times):
    # Calculate and save mutant fitnesses for bootstrapping later
    # inputs are the same as calc_fitness
    # Saves a dictionary 'data' with a list of all fitnesses for each BC and condition (across all reps)
    
    data={}
    conds = list(cond_rep_times.keys())
    BCs = np.unique(all_counts.index).tolist()
    for cond in conds:
        data[cond]={}
        for BC in BCs:
            #data[BC]={}
            #data[BC][cond] = []
            this_fit=[]
            for rep in [1,2,3]:
                timepoints = cond_rep_times[cond][rep-1]
                these_counts = all_counts[all_counts['condition']==f'{cond}{rep}']
                this_BC = these_counts[these_counts.index==BC]
                for timept in timepoints:
                    total_now = np.sum(these_counts[these_counts['time']==timept]['count'])
                    total_prev = np.sum(these_counts[these_counts['time']==timept-1]['count'])
                    now = this_BC[this_BC['time']==timept]
                    prev = this_BC[this_BC['time']==timept-1]
                    count_now = now.loc[BC,'count']
                    count_prev = prev.loc[BC,'count']
                    freq_now = count_now/total_now
                    freq_prev = count_prev/total_prev
                    if (count_now > 0) & (count_prev > 0):
                        this_fit.append(np.log(freq_now/freq_prev)+ave_mean_fit[cond][rep][timept])
                    else:
                        #print('zero')
                        break

            if len(this_fit) > 0:
                data[cond][BC] = this_fit
            else:
                data[cond][BC] = []
    
    return data


def save_fitness_skip_zeros_static(all_counts,ave_mean_fit,cond_rep_times):
    # USE THIS FN ONLY FOR STATIC MEASUREMENTS
    # Calculate and save mutant fitnesses for bootstrapping later
    # inputs are the same as calc_fitness
    # Saves a dictionary 'data' with a list of all fitnesses for each BC and condition (across all reps)
    
    data={}
    conds = list(cond_rep_times.keys())
    BCs = np.unique(all_counts.index).tolist()
    for cond in conds:
        data[cond]={}
        for BC in BCs:
            #data[BC]={}
            this_fit=[]
            for rep in [1,2,3]:
                timepoints = cond_rep_times[cond][rep-1]
                these_counts = all_counts[all_counts['condition']==f'{cond}{rep}']
                #these_counts['freq'] = these_counts['count']/np.sum(these_counts['count'])
                this_BC = these_counts[these_counts.index==BC]
                this_BC_freqs = []
                for k in range(5):
                    total = np.sum(these_counts[these_counts['time']==k]['count'])
                    this_BC_freq = this_BC[this_BC['time']==k]['count'].tolist()[0]/total
                    this_BC_freqs.append(this_BC_freq)
                
                # Make a list of steps to take between frequencies:
                steps=[]
                step=1
                if this_BC_freqs[0] != 0:
                    
                    for timept in timepoints:
                        if this_BC_freqs[timept] > 0:
                            steps.append(step)
                            step=1
                        else:
                            steps.append(0)
                            step+=1
                    
                    for timept in timepoints:
                        freq_now = this_BC_freqs[timept]
                        if freq_now > 0:
                            prev_point = steps[timept-1]
                            freq_prev = this_BC_freqs[timept-prev_point]
                            mean_fit=0
                            for k in range(prev_point):
                                # Calculate sum of mean fitness across timepoints
                                mean_fit += ave_mean_fit[cond][rep][timept-k]
                              
                            # Calculate fitness, making sure to divide by more than one step if necessary    
                            this_fit.append(np.log(freq_now/freq_prev)/prev_point + mean_fit)

            if len(this_fit) > 0:
                data[cond][BC] = this_fit
            else:
                data[cond][BC] = []
    
    return data
    
    

def calc_fitness_MLE_naive(all_counts,ave_mean_fit,cond_rep_times,rep_sep=True):
    # Calculate mutant fitnesses
    # cond_rep_times is a dictionary where each condition is a dictionary containing 
    # a list of three lists of timepoints to use with each replicate
    # If rep_sep, calculate fitness of each replicate separately

    all_BCs = []
    all_conds = []
    all_reps = []
    all_fits = []
    all_stds = []
    all_naive_std_errs = []
    conds = list(cond_rep_times.keys())
    BCs = np.unique(all_counts.index).tolist()
    for cond in conds:
        for BC in BCs:
            this_fit=[]
            for rep in [1,2,3]:
                if rep_sep:
                    this_fit=[]
                timepoints = cond_rep_times[cond][rep-1]
                these_counts = all_counts[all_counts['condition']==f'{cond}{rep}']
                this_BC = these_counts[these_counts.index==BC]
                for timept in timepoints:
                    total_now = np.sum(these_counts[these_counts['time']==timept]['count'])
                    total_prev = np.sum(these_counts[these_counts['time']==timept-1]['count'])
                    now = this_BC[this_BC['time']==timept]
                    prev = this_BC[this_BC['time']==timept-1]
                    count_now = now.loc[BC,'count']
                    count_prev = prev.loc[BC,'count']
                    freq_now = count_now/total_now
                    freq_prev = count_prev/total_prev
                    if (count_now > 0) & (count_prev > 0):
                        this_fit.append(np.log(freq_now/freq_prev)+ave_mean_fit[cond][rep][timept])
                    else:
                        #print('zero')
                        break
                if rep_sep:
                    if len(this_fit) > 0:
                        all_BCs.append(BC)
                        all_conds.append(cond)
                        all_reps.append(rep)
                        
                        data = this_fit
                        
                        #def neg_log_likelihood(params):
                        def neg_log_likelihood(params, data):
                            mean, std = params
                            likelihood = np.sum(-np.log(std) - ((data - mean)**2) / (2 * std ** 2))
                            return -likelihood
                        
                        # Initial guess for mean and std
                        initial_guess = [0, 0.1]
                        # Add bounds to the parameters (mean can be any real number; std must be > 0)
                        param_bounds = [(-np.inf, np.inf), (1e-6, np.inf)]
                        # Run the optimizer with bounds
                        result = minimize(neg_log_likelihood, initial_guess, args=(data,), method='L-BFGS-B', bounds=param_bounds)
                        # Extract the estimated parameters
                        estimated_mean, estimated_std = result.x

                            
                        naive_std_err = estimated_std/np.sqrt(len(data))
                        
                        all_fits.append(estimated_mean)
                        all_stds.append(estimated_std)
                        all_naive_std_errs.append(naive_std_err)
                        #print(std_err, naive_std_err)
                        
                    else:
                        all_BCs.append(BC)
                        all_conds.append(cond)
                        all_reps.append(rep)
                        all_fits.append(np.nan)
                        all_stds.append(np.nan)
                        all_naive_std_errs.append(np.nan)
                        
            if rep_sep==False:
                    
                if len(this_fit) > 0:
                    all_BCs.append(BC)
                    all_conds.append(cond)
                    data = this_fit
                    
                    #def neg_log_likelihood(params):
                    def neg_log_likelihood(params, data):
                        mean, std = params
                        likelihood = np.sum(-np.log(std) - ((data - mean)**2) / (2 * std**2))
                        return -likelihood  # We return the negative because 'minimize' minimizes, and we want to maximize
                    
                    # Initial guess for mean and std
                    initial_guess = [0, 0.1]
                    # Add bounds to the parameters (mean can be any real number; std must be > 0)
                    param_bounds = [(-np.inf, np.inf), (1e-6, np.inf)]
                    # Run the optimizer with bounds
                    result = minimize(neg_log_likelihood, initial_guess, args=(data,), method='L-BFGS-B', bounds=param_bounds)
                    # Extract the estimated parameters
                    estimated_mean, estimated_std = result.x
                    
                    naive_std_err = estimated_std/np.sqrt(len(data))

                    all_fits.append(estimated_mean)
                    all_stds.append(estimated_std)
                    all_naive_std_errs.append(naive_std_err)
                    #print(std_err, naive_std_err)
                    
                else:
                    all_BCs.append(BC)
                    all_conds.append(cond)
                    all_fits.append(np.nan)
                    all_stds.append(np.nan)
                    all_naive_std_errs.append(np.nan)
                
    if rep_sep:        
        data = {
            'BC' : all_BCs,
            'condition' : all_conds,
            'rep' : all_reps,
            'fitness': all_fits,
            'std': all_stds,
            'Naive_std_err': all_naive_std_errs
        }
    else:
        data = {
            'BC' : all_BCs,
            'condition' : all_conds,
            'fitness': all_fits,
            'std': all_stds,
            'Naive_std_err': all_naive_std_errs
        }
    # creating a Dataframe object 
    df_all_fit = pd.DataFrame(data) 
    df_all_fit.set_index('BC',inplace=True)
    
    return df_all_fit




def calc_fitness_MLE_naive_skip_zeros_static(all_counts,ave_mean_fit,cond_rep_times,rep_sep=True):
    # Calculate mutant fitnesses
    # ONLY USE THIS FN FOR STATIC FITNESS- it jumps timepoints that have zero freq
    # cond_rep_times is a dictionary where each condition is a dictionary containing 
    # a list of three lists of timepoints to use with each replicate
    # If rep_sep, calculate fitness of each replicate separately

    all_BCs = []
    all_conds = []
    all_reps = []
    all_fits = []
    all_stds = []
    all_naive_std_errs = []
    conds = list(cond_rep_times.keys())
    BCs = np.unique(all_counts.index).tolist()
    for cond in conds:
        for BC in BCs:
            this_fit=[]
            for rep in [1,2,3]:
                if rep_sep:
                    this_fit=[]
                timepoints = cond_rep_times[cond][rep-1]
                these_counts = all_counts[all_counts['condition']==f'{cond}{rep}']
                #these_counts['freq'] = these_counts['count']/np.sum(these_counts['count'])
                this_BC = these_counts[these_counts.index==BC]
                this_BC_freqs = []
                for k in range(5):
                    total = np.sum(these_counts[these_counts['time']==k]['count'])
                    this_BC_freq = this_BC[this_BC['time']==k]['count'].tolist()[0]/total
                    this_BC_freqs.append(this_BC_freq)
                
                # Make a list of steps to take between frequencies:
                steps=[]
                step=1
                if this_BC_freqs[0] != 0:
                    
                    for timept in timepoints:
                        if this_BC_freqs[timept] > 0:
                            steps.append(step)
                            step=1
                        else:
                            steps.append(0)
                            step+=1
                    
                    for timept in timepoints:
                        freq_now = this_BC_freqs[timept]
                        if freq_now > 0:
                            prev_point = steps[timept-1]
                            freq_prev = this_BC_freqs[timept-prev_point]
                            mean_fit=0
                            for k in range(prev_point):
                                # Calculate sum of mean fitness across timepoints
                                mean_fit += ave_mean_fit[cond][rep][timept-k]
                              
                            # Calculate fitness, making sure to divide by more than one step if necessary    
                            this_fit.append(np.log(freq_now/freq_prev)/prev_point + mean_fit)
                            
                elif sum([1 for x in this_BC_freqs if x != 0]) >= 2:
                    print(f'Warning: BC {BC} in cond {cond}{rep} not included for zero starting freq')
                
                if rep_sep:
                    if len(this_fit) > 0:
                        all_BCs.append(BC)
                        all_conds.append(cond)
                        all_reps.append(rep)
                        
                        data = this_fit
                        
                        def neg_log_likelihood(params):
                        #def neg_log_likelihood(params, data):
                            mean, std = params
                            likelihood = np.sum(-np.log(std) - ((data - mean)**2) / (2 * std ** 2))
                            return -likelihood
                        
                        # Initial guess for mean and std
                        initial_guess = [0, 0.1]
                        # Add bounds to the parameters (mean can be any real number; std must be > 0)
                        param_bounds = [(-np.inf, np.inf), (1e-6, np.inf)]
                        # Run the optimizer with bounds
                        result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B', bounds=param_bounds)
                        #result = minimize(neg_log_likelihood, initial_guess, args=(data,), method='L-BFGS-B', bounds=param_bounds)
                        # Extract the estimated parameters
                        estimated_mean, estimated_std = result.x

                            
                        naive_std_err = estimated_std/np.sqrt(len(data))
                        
                        all_fits.append(estimated_mean)
                        all_stds.append(estimated_std)
                        all_naive_std_errs.append(naive_std_err)
                        #print(std_err, naive_std_err)
                        
                    else:
                        all_BCs.append(BC)
                        all_conds.append(cond)
                        all_reps.append(rep)
                        all_fits.append(np.nan)
                        all_stds.append(np.nan)
                        all_naive_std_errs.append(np.nan)
                        
            if rep_sep==False:
                    
                if len(this_fit) > 0:
                    all_BCs.append(BC)
                    all_conds.append(cond)
                    data = this_fit
                    
                    def neg_log_likelihood(params):
                    #def neg_log_likelihood(params, data):
                        mean, std = params
                        likelihood = np.sum(-np.log(std) - ((data - mean)**2) / (2 * std**2))
                        return -likelihood  # We return the negative because 'minimize' minimizes, and we want to maximize
                    
                    # Initial guess for mean and std
                    initial_guess = [0, 0.1]
                    # Add bounds to the parameters (mean can be any real number; std must be > 0)
                    param_bounds = [(-np.inf, np.inf), (1e-6, np.inf)]
                    # Run the optimizer with bounds
                    result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B', bounds=param_bounds)
                    #result = minimize(neg_log_likelihood, initial_guess, args=(data,), method='L-BFGS-B', bounds=param_bounds)
                    # Extract the estimated parameters
                    estimated_mean, estimated_std = result.x
                    
                    naive_std_err = estimated_std/np.sqrt(len(data))

                    all_fits.append(estimated_mean)
                    all_stds.append(estimated_std)
                    all_naive_std_errs.append(naive_std_err)
                    #print(std_err, naive_std_err)
                    
                else:
                    all_BCs.append(BC)
                    all_conds.append(cond)
                    all_fits.append(np.nan)
                    all_stds.append(np.nan)
                    all_naive_std_errs.append(np.nan)
                
    if rep_sep:        
        data = {
            'BC' : all_BCs,
            'condition' : all_conds,
            'rep' : all_reps,
            'fitness': all_fits,
            'std': all_stds,
            'Naive_std_err': all_naive_std_errs
        }
    else:
        data = {
            'BC' : all_BCs,
            'condition' : all_conds,
            'fitness': all_fits,
            'std': all_stds,
            'Naive_std_err': all_naive_std_errs
        }
    # creating a Dataframe object 
    df_all_fit = pd.DataFrame(data) 
    df_all_fit.set_index('BC',inplace=True)
    
    return df_all_fit


#%% Calculate MLE fitnesses with naive std err, static:
    
# Static conditions:    
cond_rep_times_static={}
conds = ['Glu','Gal','L','high_H_H','N']

for c in conds:
    cond_rep_times_static[c] = {}
timepoints = [1,2,3,4]

for c in conds:
    cond_rep_times_static[c]=[]
    for r in range(3):
        cond_rep_times_static[c].append(timepoints)

# Get mean fitness first:
ave_mean_fit = calc_mean_fit(all_counts,cond_rep_times_static)

# replicates together:    
rep_sep=False
df_static_fit_rep_together_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit,cond_rep_times_static,rep_sep)
df_static_fit_rep_together_MLE_naive.to_csv('data/MLE_naive_calc_static_fits_across_reps.csv')

# replicates separate: 
rep_sep=True
df_static_fit_rep_sep_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit,cond_rep_times_static,rep_sep)
df_static_fit_rep_sep_MLE_naive.to_csv('data/MLE_naive_calc_static_fits_sep_reps.csv')

#%% Calculate MLE fitnesses with naive std err, static, jump timepoints with zero freq:
    
# Static conditions:    
cond_rep_times_static={}
conds = ['Glu','Gal','L','high_H_H','N']

for c in conds:
    cond_rep_times_static[c] = {}
timepoints = [1,2,3,4]

for c in conds:
    cond_rep_times_static[c]=[]
    for r in range(3):
        cond_rep_times_static[c].append(timepoints)

# Get mean fitness first:
ave_mean_fit = calc_mean_fit(all_counts,cond_rep_times_static)

# replicates together:    
rep_sep=False
#df_static_fit_rep_together_MLE_naive_skip_zeros = calc_fitness_MLE_naive_skip_zeros_static(all_counts,ave_mean_fit,cond_rep_times_static,rep_sep)
#df_static_fit_rep_together_MLE_naive_skip_zeros.to_csv('data/MLE_naive_calc_static_fits_across_reps_skip_zeros.csv')

# replicates separate: 
rep_sep=True
df_static_fit_rep_sep_MLE_naive_skip_zeros = calc_fitness_MLE_naive_skip_zeros_static(all_counts,ave_mean_fit,cond_rep_times_static,rep_sep)
df_static_fit_rep_sep_MLE_naive_skip_zeros.to_csv('data/MLE_naive_calc_static_fits_sep_reps_skip_zeros.csv')

#%% Save static fitnesses for bootstrapping later, jump timepoints with zero freq:
    
# Static conditions:    
cond_rep_times_static={}
conds = ['Glu','Gal','L','N']

for c in conds:
    cond_rep_times_static[c] = {}
timepoints = [1,2,3,4]

for c in conds:
    cond_rep_times_static[c]=[]
    for r in range(3):
        cond_rep_times_static[c].append(timepoints)

# Get mean fitness first:
ave_mean_fit = calc_mean_fit(all_counts,cond_rep_times_static)

skipz_static_data = save_fitness_skip_zeros_static(all_counts,ave_mean_fit,cond_rep_times_static)
with open('fit_lists_for_bootstrap/skipz_static_data.pkl', 'wb') as f:
    pickle.dump(skipz_static_data, f)
    
#%% Save static fitnesses for bootstrapping later, EXCEPTION: H only leaving out T4, jump timepoints with zero freq:

# Static conditions:    
cond_rep_times_static_H={}
conds = ['high_H_H']

for c in conds:
    cond_rep_times_static_H[c] = {}
timepoints = [1,2,3]

for c in conds:
    cond_rep_times_static_H[c]=[]
    for r in range(3):
        cond_rep_times_static_H[c].append(timepoints)
        
# Get mean fitness first:
ave_mean_fit_H = calc_mean_fit(all_counts,cond_rep_times_static_H)

skipz_static_data_H = save_fitness_skip_zeros_static(all_counts,ave_mean_fit_H,cond_rep_times_static_H)
with open('fit_lists_for_bootstrap/skipz_static_data_H_exception.pkl', 'wb') as f:
    pickle.dump(skipz_static_data_H, f)

#%% Calculate MLE fitnesses with naive std err, EXCEPTION: H only leaving out T4, jump timepoints with zero freq:
 
# Static conditions:    
cond_rep_times_static_H={}
conds = ['high_H_H']

for c in conds:
    cond_rep_times_static_H[c] = {}
timepoints = [1,2,3]

for c in conds:
    cond_rep_times_static_H[c]=[]
    for r in range(3):
        cond_rep_times_static_H[c].append(timepoints)
        
# Get mean fitness first:
ave_mean_fit_H = calc_mean_fit(all_counts,cond_rep_times_static_H)

# replicates together:    
rep_sep=False
df_static_fit_rep_together_MLE_naive_skip_zeros_H = calc_fitness_MLE_naive_skip_zeros_static(all_counts,ave_mean_fit_H,cond_rep_times_static_H,rep_sep)
df_static_fit_rep_together_MLE_naive_skip_zeros_H.to_csv('data/MLE_naive_calc_static_fits_across_reps_skip_zeros_H.csv')

# replicates separate: 
rep_sep=True
df_static_fit_rep_sep_MLE_naive_skip_zeros_H = calc_fitness_MLE_naive_skip_zeros_static(all_counts,ave_mean_fit_H,cond_rep_times_static_H,rep_sep)
df_static_fit_rep_sep_MLE_naive_skip_zeros_H.to_csv('data/MLE_naive_calc_static_fits_sep_reps_skip_zeros_H.csv')


#%%
# Fluctuating conditions, mean fitness:
    
fluct_conds = ['Glu-Gal','Glu-L','high_H_Glu-H','Glu-N','high_H_H-N','Gal-L','high_H_Gal-H','Gal-N','high_H_L-H','L-N']

#s1:
cond_rep_times_fluct_comp1={}
for c in fluct_conds:
    cond_rep_times_fluct_comp1[c] = {}
comp1_timepoints = [2,4]

for c in fluct_conds:
    cond_rep_times_fluct_comp1[c]=[]
    for r in range(3):
        cond_rep_times_fluct_comp1[c].append(comp1_timepoints)
# Get mean fitness first:
ave_mean_fit_fluct_comp1 = calc_mean_fit(all_counts,cond_rep_times_fluct_comp1)
        
#s2, without T0:
cond_rep_times_fluct_comp2_withoutT0={}
for c in fluct_conds:
    cond_rep_times_fluct_comp2_withoutT0[c] = {}
comp2_timepoints = [3,5]

for c in fluct_conds:
    cond_rep_times_fluct_comp2_withoutT0[c]=[]
    for r in range(3):
        cond_rep_times_fluct_comp2_withoutT0[c].append(comp2_timepoints)
# Get mean fitness first:
ave_mean_fit_fluct_comp2_withoutT0 = calc_mean_fit(all_counts,cond_rep_times_fluct_comp2_withoutT0)
        
#s2, with T0:
cond_rep_times_fluct_comp2_withT0={}
for c in fluct_conds:
    cond_rep_times_fluct_comp2_withT0[c] = {}
comp2_timepoints = [1,3,5]

for c in fluct_conds:
    cond_rep_times_fluct_comp2_withT0[c]=[]
    for r in range(3):
        cond_rep_times_fluct_comp2_withT0[c].append(comp2_timepoints)
# Get mean fitness first:
ave_mean_fit_fluct_comp2_withT0 = calc_mean_fit(all_counts,cond_rep_times_fluct_comp2_withT0)

#s2, with T0, without T5 for Lac/H, Glu/H, and Gal/Lac rep 3:
cond_rep_times_fluct_comp2_withT0_exceptions={}
for c in ['high_H_Glu-H','Gal-L','high_H_L-H']:
    cond_rep_times_fluct_comp2_withT0_exceptions[c] = {}
comp2_timepoints_exceptions = [1,3]

for c in ['high_H_Glu-H','Gal-L','high_H_L-H']:
    cond_rep_times_fluct_comp2_withT0_exceptions[c]=[]
    for r in range(3):
        cond_rep_times_fluct_comp2_withT0_exceptions[c].append(comp2_timepoints_exceptions)
cond_rep_times_fluct_comp2_withT0_exceptions['Gal-L'][0]=[1,3,5]
cond_rep_times_fluct_comp2_withT0_exceptions['Gal-L'][1]=[1,3,5]
# Get mean fitness first:
ave_mean_fit_fluct_comp2_withT0_exceptions = calc_mean_fit(all_counts,cond_rep_times_fluct_comp2_withT0_exceptions)

#%% Fluctuating conditions, fitness of mutants:
        
# replicates separate: 
rep_sep=True
df_fluct_comp1_rep_sep = calc_fitness(all_counts,ave_mean_fit_fluct_comp1,cond_rep_times_fluct_comp1,rep_sep)
df_fluct_comp1_rep_sep.to_csv('data/calc_fluct_fits_comp1_sep_reps.csv')

df_fluct_comp2_withoutT0_rep_sep = calc_fitness(all_counts,ave_mean_fit_fluct_comp2_withoutT0,cond_rep_times_fluct_comp2_withoutT0,rep_sep)
df_fluct_comp2_withoutT0_rep_sep.to_csv('data/calc_fluct_fits_comp2_withoutT0_sep_reps.csv')

df_fluct_comp2_withT0_rep_sep = calc_fitness(all_counts,ave_mean_fit_fluct_comp2_withT0,cond_rep_times_fluct_comp2_withT0,rep_sep)
df_fluct_comp2_withT0_rep_sep.to_csv('data/calc_fluct_fits_comp2_withT0_sep_reps.csv')



#%% Static conditions MEM CONTROL: treat as if fluctuating

#%% Mean fitness first:

static_fluct_conds = ['Glu','Gal','L','high_H_H','N']

#s1:
static_cond_rep_times_fluct_comp1={}
for c in static_fluct_conds:
    static_cond_rep_times_fluct_comp1[c] = {}
static_comp1_timepoints = [1,3]

for c in static_fluct_conds:
    static_cond_rep_times_fluct_comp1[c]=[]
    for r in range(3):
        static_cond_rep_times_fluct_comp1[c].append(static_comp1_timepoints)
# Get mean fitness first:
static_ave_mean_fit_fluct_comp1 = calc_mean_fit(all_counts,static_cond_rep_times_fluct_comp1)
        
#s2:
static_cond_rep_times_fluct_comp2={}
for c in static_fluct_conds:
    static_cond_rep_times_fluct_comp2[c] = {}
static_comp2_timepoints = [2,4]

for c in static_fluct_conds:
    static_cond_rep_times_fluct_comp2[c]=[]
    for r in range(3):
        static_cond_rep_times_fluct_comp2[c].append(static_comp2_timepoints)
# Get mean fitness first:
static_ave_mean_fit_fluct_comp2 = calc_mean_fit(all_counts,static_cond_rep_times_fluct_comp2)
        
#%% Fitness of mutants:
        
# replicates separate: 
rep_sep=True
static_df_fluct_comp1_rep_sep = calc_fitness(all_counts,static_ave_mean_fit_fluct_comp1,static_cond_rep_times_fluct_comp1,rep_sep)
static_df_fluct_comp1_rep_sep.to_csv('data/static_calc_fluct_fits_comp1_sep_reps.csv')

static_df_fluct_comp2_rep_sep = calc_fitness(all_counts,static_ave_mean_fit_fluct_comp2,static_cond_rep_times_fluct_comp2,rep_sep)
static_df_fluct_comp2_rep_sep.to_csv('data/static_calc_fluct_fits_comp2_sep_reps.csv')


#%% Fluctuating conditions REGRESSION TO MEAN CONTROL: 3rd and 6th timepoints only

#%% Mean fitness first:

fluct_conds = ['Glu-Gal','Glu-L','high_H_Glu-H','Glu-N','high_H_H-N','Gal-L','high_H_Gal-H','Gal-N','high_H_L-H','L-N']

#s1:
cond_rep_times_fluct_comp1_R2M={}
for c in fluct_conds:
    cond_rep_times_fluct_comp1_R2M[c] = {}
comp1_timepoints_R2M = [2]

for c in fluct_conds:
    cond_rep_times_fluct_comp1_R2M[c]=[]
    for r in range(3):
        cond_rep_times_fluct_comp1_R2M[c].append(comp1_timepoints_R2M)
# Get mean fitness first:
ave_mean_fit_fluct_comp1_R2M = calc_mean_fit(all_counts,cond_rep_times_fluct_comp1_R2M)
        
#s2, without T0:
cond_rep_times_fluct_comp2_R2M={}
for c in fluct_conds:
    cond_rep_times_fluct_comp2_R2M[c] = {}
comp2_timepoints_R2M = [5]

for c in fluct_conds:
    cond_rep_times_fluct_comp2_R2M[c]=[]
    for r in range(3):
        cond_rep_times_fluct_comp2_R2M[c].append(comp2_timepoints_R2M)
# Get mean fitness first:
ave_mean_fit_fluct_comp2_R2M = calc_mean_fit(all_counts,cond_rep_times_fluct_comp2_R2M)

#%% Fluctuating conditions R2M control, fitness of mutants:
        
# replicates separate: 
rep_sep=True
df_fluct_comp1_rep_sep_R2M = calc_fitness(all_counts,ave_mean_fit_fluct_comp1_R2M,cond_rep_times_fluct_comp1_R2M,rep_sep)
df_fluct_comp1_rep_sep_R2M.to_csv('data/calc_fluct_fits_comp1_sep_reps_R2M_control.csv')

df_fluct_comp2_rep_sep_R2M = calc_fitness(all_counts,ave_mean_fit_fluct_comp2_R2M,cond_rep_times_fluct_comp2_R2M,rep_sep)
df_fluct_comp2_rep_sep_R2M.to_csv('data/calc_fluct_fits_comp2_sep_reps_R2M_control.csv')


#%%


#%% Calculate MLE fitnesses with naive std err, fluctuating:

# replicates together:    
rep_sep=False
df_fluct_comp1_rep_together_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit_fluct_comp1,cond_rep_times_fluct_comp1,rep_sep)
df_fluct_comp1_rep_together_MLE_naive.to_csv('data/MLE_naive_fluct_fits_comp1_across_reps.csv')

df_fluct_comp2_withoutT0_rep_together_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit_fluct_comp2_withoutT0,cond_rep_times_fluct_comp2_withoutT0,rep_sep)
df_fluct_comp2_withoutT0_rep_together_MLE_naive.to_csv('data/MLE_naive_fluct_fits_comp2_withoutT0_across_reps.csv')

df_fluct_comp2_withT0_rep_together_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit_fluct_comp2_withT0,cond_rep_times_fluct_comp2_withT0,rep_sep)
df_fluct_comp2_withT0_rep_together_MLE_naive.to_csv('data/MLE_naive_fluct_fits_comp2_withT0_across_reps.csv')

# replicates separate: 
rep_sep=True
df_fluct_comp1_rep_sep_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit_fluct_comp1,cond_rep_times_fluct_comp1,rep_sep)
df_fluct_comp1_rep_sep_MLE_naive.to_csv('data/MLE_naive_fluct_fits_comp1_sep_reps.csv')

df_fluct_comp2_withoutT0_rep_sep_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit_fluct_comp2_withoutT0,cond_rep_times_fluct_comp2_withoutT0,rep_sep)
df_fluct_comp2_withoutT0_rep_sep_MLE_naive.to_csv('data/MLE_naive_fluct_fits_comp2_withoutT0_sep_reps.csv')

df_fluct_comp2_withT0_rep_sep_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit_fluct_comp2_withT0,cond_rep_times_fluct_comp2_withT0,rep_sep)
df_fluct_comp2_withT0_rep_sep_MLE_naive.to_csv('data/MLE_naive_fluct_fits_comp2_withT0_sep_reps.csv')


#%% Calculate MLE fitnesses with naive std err, fluctuating EXCEPTIONS (timepoints dropped due to high pool fraction or missing data):

# replicates together:    
rep_sep=False
df_fluct_comp2_withT0_exceptions_rep_together_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit_fluct_comp2_withT0_exceptions,cond_rep_times_fluct_comp2_withT0_exceptions,rep_sep)
df_fluct_comp2_withT0_exceptions_rep_together_MLE_naive.to_csv('data/MLE_naive_fluct_fits_comp2_withT0_exceptions_across_reps.csv')

# replicates separate: 
rep_sep=True
df_fluct_comp2_withT0_exceptions_rep_together_MLE_naive = calc_fitness_MLE_naive(all_counts,ave_mean_fit_fluct_comp2_withT0_exceptions,cond_rep_times_fluct_comp2_withT0_exceptions,rep_sep)
df_fluct_comp2_withT0_exceptions_rep_together_MLE_naive.to_csv('data/MLE_naive_fluct_fits_comp2_withT0_exceptions_sep_reps.csv')


#%% Save fluctuating fitnesses for bootstrapping later:
    
fluct_comp1_data = save_fitness(all_counts,ave_mean_fit_fluct_comp1,cond_rep_times_fluct_comp1)
with open('fit_lists_for_bootstrap/fluct_comp1_data.pkl', 'wb') as f:
    pickle.dump(fluct_comp1_data, f)
    
fluct_comp2_data_withoutT0 = save_fitness(all_counts,ave_mean_fit_fluct_comp2_withoutT0,cond_rep_times_fluct_comp2_withoutT0)
with open('fit_lists_for_bootstrap/fluct_comp2_data_withoutT0.pkl', 'wb') as f:
    pickle.dump(fluct_comp2_data_withoutT0, f)
    
fluct_comp2_data_withT0 = save_fitness(all_counts,ave_mean_fit_fluct_comp2_withT0,cond_rep_times_fluct_comp2_withT0)
with open('fit_lists_for_bootstrap/fluct_comp2_data_withT0.pkl', 'wb') as f:
    pickle.dump(fluct_comp2_data_withT0, f)
    
#%% Save fluctuating fitnesses for bootstrapping later, EXCEPTIONS:

fluct_comp2_data_withT0_exceptions = save_fitness(all_counts,ave_mean_fit_fluct_comp2_withT0_exceptions,cond_rep_times_fluct_comp2_withT0_exceptions)
with open('fit_lists_for_bootstrap/fluct_comp2_data_withT0_exceptions.pkl', 'wb') as f:
    pickle.dump(fluct_comp2_data_withT0_exceptions, f)
    
    
    
    
    