#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:24:06 2023

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

#%% Functions:

def consolidate_bootstrap(bootstrap_data1,bootstrap_data2):
    # Consolidate data from two bootstrap runs
    conds = list(bootstrap_data1.keys())
    BCs = list(bootstrap_data1[conds[0]].keys())
    bootstrap_comb = {}
    for c in conds:
        bootstrap_comb[c] = {}
        for BC in BCs:
            bootstrap_comb[c][BC] = bootstrap_data1[c][BC] + bootstrap_data2[c][BC]
    return(bootstrap_comb)
    

def add_bootstrap_mean_SEM(df,conds,conds_dict,bootstrap_data):
    # Add bootstrap mean, SEM, and SEM across replicates to dataframe
    BCs = df.index.tolist()
    df_bootstrap = df.copy()
    for cond in conds:
        cond_name = conds_dict[cond]
        df_bootstrap[f'{cond_name}_fitness_bootstrap'] = np.nan
        df_bootstrap[f'{cond_name}_SEM_bootstrap'] = np.nan
        df_bootstrap[f'{cond_name}_fitness_across_reps'] = np.nan
        df_bootstrap[f'{cond_name}_SEM_across_reps'] = np.nan
        for BC in BCs:
            if BC in list(bootstrap_data[cond].keys()):
                these_fits = bootstrap_data[cond][BC]
                if len(these_fits)>0:
                    mean_fit = np.mean(these_fits)
                    SEM_fit = np.std(these_fits)
                    df_bootstrap.loc[BC,f'{cond_name}_fitness_bootstrap'] = mean_fit
                    df_bootstrap.loc[BC,f'{cond_name}_SEM_bootstrap'] = SEM_fit
    for cond in conds:
        cond_name = conds_dict[cond]
        for BC in BCs:
            num_non_nan = df_bootstrap.loc[BC,[f'{cond_name}_{r}_fitness' for \
                                                               r in [1,2,3] ]].count()
            if num_non_nan > 1:    
                df_bootstrap.loc[BC,f'{cond_name}_fitness_across_reps'] = \
                    np.nanmean(df_bootstrap.loc[BC,[f'{cond_name}_{r}_fitness' for \
                    r in [1,2,3] ]].tolist())
                df_bootstrap.loc[BC,f'{cond_name}_SEM_across_reps'] = \
                    np.nanstd(df_bootstrap.loc[BC,[f'{cond_name}_{r}_fitness' for \
                    r in [1,2,3] ]].tolist(),ddof=1)/np.sqrt(num_non_nan)
    return(df_bootstrap)


def add_fluct_bootstrap_mean_SEM(df,conds,conds_dict,comp_dict,bootstrap_data,first_or_second):
    # Add bootstrap mean, SEM, and SEM across replicates to dataframe
    BCs = df.index.tolist()
    df_bootstrap = df.copy()
    for cond in conds:
        cond_name = conds_dict[cond]
        if first_or_second == 'first':
            comp_name = comp_dict[cond_name][0]
        elif first_or_second == 'second':
            comp_name = comp_dict[cond_name][1]
        df_bootstrap[f'{comp_name}_{cond_name}_fitness_bootstrap'] = np.nan
        df_bootstrap[f'{comp_name}_{cond_name}_SEM_bootstrap'] = np.nan
        df_bootstrap[f'{comp_name}_{cond_name}_fitness_across_reps'] = np.nan
        df_bootstrap[f'{comp_name}_{cond_name}_SEM_across_reps'] = np.nan
        for BC in BCs:
            if BC in list(bootstrap_data[cond].keys()):
                these_fits = bootstrap_data[cond][BC]
                if len(these_fits)>0:
                    mean_fit = np.mean(these_fits)
                    SEM_fit = np.std(these_fits)
                    df_bootstrap.loc[BC,f'{comp_name}_{cond_name}_fitness_bootstrap'] = mean_fit
                    df_bootstrap.loc[BC,f'{comp_name}_{cond_name}_SEM_bootstrap'] = SEM_fit
    for cond in conds:
        cond_name = conds_dict[cond]
        if first_or_second == 'first':
            comp_name = comp_dict[cond_name][0]
        elif first_or_second == 'second':
            comp_name = comp_dict[cond_name][1]
        for BC in BCs:
            num_non_nan = df_bootstrap.loc[BC,[f'{comp_name}_{cond_name}_{r}_fitness' for \
                                                               r in [1,2,3] ]].count()
            if num_non_nan > 1:    
                df_bootstrap.loc[BC,f'{comp_name}_{cond_name}_fitness_across_reps'] = \
                    np.nanmean(df_bootstrap.loc[BC,[f'{comp_name}_{cond_name}_{r}_fitness' for \
                    r in [1,2,3] ]].tolist())
                df_bootstrap.loc[BC,f'{comp_name}_{cond_name}_SEM_across_reps'] = \
                    np.nanstd(df_bootstrap.loc[BC,[f'{comp_name}_{cond_name}_{r}_fitness' for \
                    r in [1,2,3] ]].tolist(),ddof=1)/np.sqrt(num_non_nan)
    return(df_bootstrap)

#%% Import and consolidate:
    
non_neutralAB_fitness_withT0_with_reps = pd.read_csv('../formatted_data/non_neutralAB_fitness_withreps_withT0.csv',index_col=0)

non_neutralAB_fitness_withoutT0_with_reps = pd.read_csv('../formatted_data/non_neutralAB_fitness_withreps_withoutT0.csv',index_col=0)

non_neutralAB_fitness_withT0_with_reps_exceptions = pd.read_csv('../formatted_data/non_neutralAB_fitness_withreps_withT0_exceptions.csv',index_col=0)

non_neutralAB_fitness_withoutT0_with_reps_exceptions = pd.read_csv('../formatted_data/non_neutralAB_fitness_withreps_withoutT0_exceptions.csv',index_col=0)
    
with open('bootstrap_data/bootstrap_skipz_static_data_Sept4.pkl', 'rb') as f:
    bootstrap_skipz_static_data_Sept4 = pickle.load(f)
with open('bootstrap_data/bootstrap_skipz_static_data_Sept5.pkl', 'rb') as f:
    bootstrap_skipz_static_data_Sept5 = pickle.load(f)
bootstrap_skipz_static_data = consolidate_bootstrap(bootstrap_skipz_static_data_Sept4,bootstrap_skipz_static_data_Sept5)
 
with open('bootstrap_data/bootstrap_skipz_static_data_H_exception_Sept4.pkl', 'rb') as f:
    bootstrap_skipz_static_data_H_exception_Sept4 = pickle.load(f)
with open('bootstrap_data/bootstrap_skipz_static_data_H_exception_Sept5.pkl', 'rb') as f:
    bootstrap_skipz_static_data_H_exception_Sept5 = pickle.load(f)
bootstrap_skipz_static_data_H_exception = consolidate_bootstrap(bootstrap_skipz_static_data_H_exception_Sept4,
                                                                bootstrap_skipz_static_data_H_exception_Sept5)
with open('bootstrap_data/bootstrap_skipz_static_data_H_exception_Sept6.pkl', 'rb') as f:
    bootstrap_skipz_static_data_H_exception_Sept6 = pickle.load(f)
bootstrap_skipz_static_data_H_exception = consolidate_bootstrap(bootstrap_skipz_static_data_H_exception,bootstrap_skipz_static_data_H_exception_Sept6)
   
    
with open('bootstrap_data/bootstrap_fluct_comp1_data_Sept4.pkl', 'rb') as f:
    bootstrap_fluct_comp1_data_Sept4 = pickle.load(f)
with open('bootstrap_data/bootstrap_fluct_comp1_data_Sept5.pkl', 'rb') as f:
    bootstrap_fluct_comp1_data_Sept5 = pickle.load(f)
bootstrap_fluct_comp1_data = consolidate_bootstrap(bootstrap_fluct_comp1_data_Sept4,bootstrap_fluct_comp1_data_Sept5)
    
with open('bootstrap_data/bootstrap_fluct_comp2_data_withT0_Sept4.pkl', 'rb') as f:
    bootstrap_fluct_comp2_data_withT0_Sept4 = pickle.load(f)
with open('bootstrap_data/bootstrap_fluct_comp2_data_withT0_Sept5.pkl', 'rb') as f:
    bootstrap_fluct_comp2_data_withT0_Sept5 = pickle.load(f)
bootstrap_fluct_comp2_data_withT0 = consolidate_bootstrap(bootstrap_fluct_comp2_data_withT0_Sept4,
                                                          bootstrap_fluct_comp2_data_withT0_Sept5)
    
with open('bootstrap_data/bootstrap_fluct_comp2_data_withT0_exceptions_Sept4.pkl', 'rb') as f:
    bootstrap_fluct_comp2_data_withT0_exceptions_Sept4 = pickle.load(f)
with open('bootstrap_data/bootstrap_fluct_comp2_data_withT0_exceptions_Sept5.pkl', 'rb') as f:
    bootstrap_fluct_comp2_data_withT0_exceptions_Sept5 = pickle.load(f)
bootstrap_fluct_comp2_data_withT0_exceptions = consolidate_bootstrap(bootstrap_fluct_comp2_data_withT0_exceptions_Sept4,
                                                          bootstrap_fluct_comp2_data_withT0_exceptions_Sept5)
    
with open('bootstrap_data/bootstrap_fluct_comp2_data_withoutT0_Sept4.pkl', 'rb') as f:
    bootstrap_fluct_comp2_data_withoutT0_Sept4 = pickle.load(f)
with open('bootstrap_data/bootstrap_fluct_comp2_data_withoutT0_Sept5.pkl', 'rb') as f:
    bootstrap_fluct_comp2_data_withoutT0_Sept5 = pickle.load(f)
bootstrap_fluct_comp2_data_withoutT0 = consolidate_bootstrap(bootstrap_fluct_comp2_data_withoutT0_Sept4,
                                                          bootstrap_fluct_comp2_data_withoutT0_Sept5)
    
#%%

static_conds = ['Glu','Gal','L','N']
static_exc_cond = ['high_H_H']
fluct_conds = ['Gal-N','Glu-Gal','Glu-L','Glu-N','high_H_Gal-H','high_H_H-N','L-N']
fluct_exc_conds = ['Gal-L','high_H_Glu-H','high_H_L-H']

static_dict = {'Glu':'Glu','Gal':'Gal','L':'Lac','high_H_H':'H2O2','N':'NaCl'}
fluct_dict = {'Glu-Gal':'Glu/Gal','Glu-L':'Glu/Lac','high_H_Glu-H':'Glu/H2O2',
              'Glu-N':'Glu/NaCl','high_H_H-N':'H2O2/NaCl','Gal-L':'Gal/Lac',
              'high_H_Gal-H':'Gal/H2O2','Gal-N':'Gal/NaCl','high_H_L-H':'Lac/H2O2',
              'L-N':'Lac/NaCl'}
fluct_comp_dict = {'Glu/Gal':['Glu','Gal'],'Glu/Lac':['Glu','Lac'],'Glu/H2O2':['Glu','H2O2'],
                   'Glu/NaCl':['Glu','NaCl'],'H2O2/NaCl':['H2O2','NaCl'],'Gal/Lac':['Gal','Lac'],
                   'Gal/H2O2':['Gal','H2O2'],'Gal/NaCl':['Gal','NaCl'],
                   'Lac/H2O2':['Lac','H2O2'],'Lac/NaCl':['Lac','NaCl']}
fluct_comp_dict_names = {'Glu/Gal':['Glu','Gal'],'Glu/Lac':['Glu','L'],'Glu/H2O2':['Glu','high_H_H'],
                   'Glu/NaCl':['Glu','N'],'H2O2/NaCl':['high_H','N'],'Gal/Lac':['Gal','L'],
                   'Gal/H2O2':['Gal','high_H_H'],'Gal/NaCl':['Gal','N'],
                   'Lac/H2O2':['L','high_H_H'],'Lac/NaCl':['L','N']}

bootstrap_withT0_with_reps = add_bootstrap_mean_SEM(non_neutralAB_fitness_withT0_with_reps_exceptions,
                                                    static_conds,static_dict,bootstrap_skipz_static_data)
bootstrap_withT0_with_reps_H = add_bootstrap_mean_SEM(bootstrap_withT0_with_reps,
                                                      static_exc_cond,static_dict,bootstrap_skipz_static_data_H_exception)
bootstrap_withT0_with_reps_comp1 = add_fluct_bootstrap_mean_SEM(bootstrap_withT0_with_reps_H,fluct_conds+fluct_exc_conds,
                                                                fluct_dict,fluct_comp_dict,bootstrap_fluct_comp1_data,'first')
bootstrap_withT0_with_reps_comps = add_fluct_bootstrap_mean_SEM(bootstrap_withT0_with_reps_comp1,fluct_conds,fluct_dict,
                                                                fluct_comp_dict,bootstrap_fluct_comp2_data_withT0,'second')
bootstrap_withT0_with_reps_comps_exceptions = add_fluct_bootstrap_mean_SEM(bootstrap_withT0_with_reps_comps,fluct_exc_conds,fluct_dict,
                                                                           fluct_comp_dict,bootstrap_fluct_comp2_data_withT0_exceptions,'second')
bootstrap_withT0_with_reps_comps_exceptions.to_csv('../formatted_data/bootstrapped_non_neutralAB_fitness_withT0_with_reps_exceptions.csv')


bootstrap_withoutT0_with_reps_comps = add_fluct_bootstrap_mean_SEM(bootstrap_withT0_with_reps_comp1,fluct_conds,fluct_dict,
                                                                fluct_comp_dict,bootstrap_fluct_comp2_data_withoutT0,'second')
bootstrap_withoutT0_with_reps_comps_exceptions = add_fluct_bootstrap_mean_SEM(bootstrap_withoutT0_with_reps_comps,fluct_exc_conds,fluct_dict,
                                                                           fluct_comp_dict,bootstrap_fluct_comp2_data_withT0_exceptions,'second')
bootstrap_withoutT0_with_reps_comps_exceptions.to_csv('../formatted_data/bootstrapped_non_neutralAB_fitness_withoutT0_with_reps_exceptions.csv')
    
#%% Plot
# Static conds:
    
df_plot = bootstrap_withT0_with_reps_comps_exceptions


for cond in static_conds+static_exc_cond:
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    cond_name = static_dict[cond]
    plt.scatter(df_plot[f'{cond_name}_fitness'],
                df_plot[f'{cond_name}_fitness_bootstrap'],
                alpha=0.2)
    plt.title(cond_name)
    plt.xlabel('Overall mean fitness')
    plt.ylabel('Bootstrap mean fitness')
    plt.xscale('log')
    plt.yscale('log')
    xpoints = plt.xlim()
    plt.plot(xpoints,xpoints,'k--') # identity line
    
    plt.subplot(1,3,2)
    cond_name = static_dict[cond]
    plt.scatter(df_plot[f'{cond_name}_fitness'],
                df_plot[f'{cond_name}_fitness_across_reps'],
                alpha=0.2)
    plt.title(cond_name)
    plt.xlabel('Overall mean fitness')
    plt.ylabel('Mean of rep fitness')
    plt.xscale('log')
    plt.yscale('log')
    xpoints = plt.xlim()
    plt.plot(xpoints,xpoints,'k--') # identity line
    
    plt.subplot(1,3,3)
    cond_name = static_dict[cond]
    plt.scatter(df_plot[f'{cond_name}_fitness_bootstrap'],
                df_plot[f'{cond_name}_fitness_across_reps'],
                alpha=0.2)
    plt.title(cond_name)
    plt.xlabel('Bootstrap mean fitness')
    plt.ylabel('Mean of rep fitness')
    plt.xscale('log')
    plt.yscale('log')
    xpoints = plt.xlim()
    plt.plot(xpoints,xpoints,'k--') # identity line
    
    plt.tight_layout()
    

    
for cond in static_conds+static_exc_cond:
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    cond_name = static_dict[cond]
    plt.scatter(df_plot[f'{cond_name}_naive_SEM'],
                df_plot[f'{cond_name}_SEM_bootstrap'],
                alpha=0.2)
    plt.title(cond_name)
    plt.xlabel('Naive SEM')
    plt.ylabel('Bootstrap SEM')
    plt.xscale('log')
    plt.yscale('log')
    xpoints = plt.xlim()
    plt.plot(xpoints,xpoints,'k--') # identity line
    
    plt.subplot(1,3,2)
    cond_name = static_dict[cond]
    plt.scatter(df_plot[f'{cond_name}_naive_SEM'],
                df_plot[f'{cond_name}_SEM_across_reps'],
                alpha=0.2)
    plt.title(cond_name)
    plt.xlabel('Naive SEM')
    plt.ylabel('Replicate SEM')
    plt.xscale('log')
    plt.yscale('log')
    xpoints = plt.xlim()
    plt.plot(xpoints,xpoints,'k--') # identity line
    
    plt.subplot(1,3,3)
    cond_name = static_dict[cond]
    plt.scatter(df_plot[f'{cond_name}_SEM_bootstrap'],
                df_plot[f'{cond_name}_SEM_across_reps'],
                alpha=0.2)
    plt.title(cond_name)
    plt.xlabel('Bootstrap SEM')
    plt.ylabel('Replicate SEM')
    plt.xscale('log')
    plt.yscale('log')
    xpoints = plt.xlim()
    plt.plot(xpoints,xpoints,'k--') # identity line
    
    plt.tight_layout()
    
#%% Plot fluct conds:

for cond in fluct_conds+fluct_exc_conds:
    for c in [0,1]:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        cond_name = fluct_dict[cond]
        comp_name = fluct_comp_dict[cond_name][c]
        plt.scatter(df_plot[f'{comp_name}_{cond_name}_fitness'],
                    df_plot[f'{comp_name}_{cond_name}_fitness_bootstrap'],
                    alpha=0.2)
        plt.title(f'{comp_name}_{cond_name}')
        plt.xlabel('Overall mean fitness')
        plt.ylabel('Bootstrap mean fitness')
        plt.xscale('log')
        plt.yscale('log')
        xpoints = plt.xlim()
        plt.plot(xpoints,xpoints,'k--') # identity line
        
        plt.subplot(1,3,2)
        cond_name = fluct_dict[cond]
        comp_name = fluct_comp_dict[cond_name][c]
        plt.scatter(df_plot[f'{comp_name}_{cond_name}_fitness'],
                    df_plot[f'{comp_name}_{cond_name}_fitness_across_reps'],
                    alpha=0.2)
        plt.title(f'{comp_name}_{cond_name}')
        plt.xlabel('Overall mean fitness')
        plt.ylabel('Mean of rep fitness')
        plt.xscale('log')
        plt.yscale('log')
        xpoints = plt.xlim()
        plt.plot(xpoints,xpoints,'k--') # identity line
        
        plt.subplot(1,3,3)
        cond_name = fluct_dict[cond]
        comp_name = fluct_comp_dict[cond_name][c]
        plt.scatter(df_plot[f'{comp_name}_{cond_name}_fitness_bootstrap'],
                    df_plot[f'{comp_name}_{cond_name}_fitness_across_reps'],
                    alpha=0.2)
        plt.title(f'{comp_name}_{cond_name}')
        plt.xlabel('Bootstrap mean fitness')
        plt.ylabel('Mean of rep fitness')
        plt.xscale('log')
        plt.yscale('log')
        xpoints = plt.xlim()
        plt.plot(xpoints,xpoints,'k--') # identity line
        
        plt.tight_layout()
        
        
for cond in fluct_conds+fluct_exc_conds:
    for c in [0,1]:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        cond_name = fluct_dict[cond]
        comp_name = fluct_comp_dict[cond_name][c]
        plt.scatter(df_plot[f'{comp_name}_{cond_name}_naive_SEM'],
                    df_plot[f'{comp_name}_{cond_name}_SEM_bootstrap'],
                    alpha=0.2)
        plt.title(f'{comp_name}_{cond_name}')
        plt.xlabel('Naive SEM')
        plt.ylabel('Bootstrap SEM')
        plt.xscale('log')
        plt.yscale('log')
        xpoints = plt.xlim()
        plt.plot(xpoints,xpoints,'k--') # identity line
        
        plt.subplot(1,3,2)
        cond_name = fluct_dict[cond]
        comp_name = fluct_comp_dict[cond_name][c]
        plt.scatter(df_plot[f'{comp_name}_{cond_name}_naive_SEM'],
                    df_plot[f'{comp_name}_{cond_name}_SEM_across_reps'],
                    alpha=0.2)
        plt.title(f'{comp_name}_{cond_name}')
        plt.xlabel('Naive SEM')
        plt.ylabel('Replicate SEM')
        plt.xscale('log')
        plt.yscale('log')
        xpoints = plt.xlim()
        plt.plot(xpoints,xpoints,'k--') # identity line
        
        plt.subplot(1,3,3)
        cond_name = fluct_dict[cond]
        comp_name = fluct_comp_dict[cond_name][c]
        plt.scatter(df_plot[f'{comp_name}_{cond_name}_SEM_bootstrap'],
                    df_plot[f'{comp_name}_{cond_name}_SEM_across_reps'],
                    alpha=0.2)
        plt.title(f'{comp_name}_{cond_name}')
        plt.xlabel('Bootstrap SEM')
        plt.ylabel('Replicate SEM')
        plt.xscale('log')
        plt.yscale('log')
        xpoints = plt.xlim()
        plt.plot(xpoints,xpoints,'k--') # identity line
        
        plt.tight_layout()