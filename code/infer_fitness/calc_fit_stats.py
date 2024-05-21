#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:01:31 2023

@author: clare
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.spatial import distance
from matplotlib import colors
import scipy
import sys
import os
from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.decomposition import PCA
from ast import literal_eval
from Bio import SeqIO
from itertools import combinations

sns.set_color_codes()
sns.set_style('white')
sns.set_style('ticks')

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %% Functions
def calc_mean_diff(fitness_df,conds):
# Compute mean fitness difference:
    diff_df = pd.DataFrame()
    diff_df.index = fitness_df.index
    diff_df['Source_Env'] = fitness_df['Source_Env']
    #conds = ['Glu','Gal','Lac','H2O2','NaCl']
    diff_cols = []
    
    for combination in itertools.combinations(conds, 2):
        diff_cols.append(f'{combination[0]}/{combination[1]}_diff')
        diff_df[f'{combination[0]}/{combination[1]}_diff'] = np.abs(fitness_df[f'{combination[0]}_fitness']-fitness_df[f'{combination[1]}_fitness'])
    
    diff_df['Mean_Diff'] = np.nan
    diff_df['Mean_Diff'] = diff_df[diff_cols].mean(axis=1) # Mean fitness difference across all pairs of conds, NaNs ignored
    
    for combination in itertools.combinations(conds, 2):
        this_comb = f'{combination[0]}/{combination[1]}'
        diff_df[f'Mean_Diff_Excl_{this_comb}'] = np.nan
        diff_cols_excl = [x for x in diff_cols if x != f'{this_comb}_diff'] # Exclude only the relevant pair of conds from this calculation
        diff_df[f'Mean_Diff_Excl_{this_comb}'] = diff_df[diff_cols_excl].mean(axis=1)
        
    for combination in itertools.combinations(conds, 2):
        this_comb = f'{combination[0]}/{combination[1]}'
        first_env = combination[0]
        sec_env = combination[1]
        diff_df[f'Mean_Diff_Excl_All_{this_comb}'] = np.nan
        # Exclude any pair of conds containing either component of this pair from this calculation:
        diff_cols_excl_all = [x for x in diff_cols if first_env not in x and sec_env not in x] 
        diff_df[f'Mean_Diff_Excl_All_{this_comb}'] = diff_df[diff_cols_excl_all].mean(axis=1)
    
    diff_df['Overall_Fit'] = 0
    diff_df['Overall_Tradeoff'] = 0
    for c in conds:
        diff_df['Overall_Fit'] += np.exp(fitness_df[f'{c}_fitness'])
    
    diff_df['Sum_Fit'] = diff_df['Overall_Fit'].copy()
    diff_df['Overall_Fit'] = np.log(diff_df['Overall_Fit']/len(conds)) # Warning: NaNs could be mishandled
        
    for c in conds:
        diff_df['Overall_Tradeoff'] += (np.exp(fitness_df[f'{c}_fitness'])/diff_df['Sum_Fit'] - 1/len(conds))**2 # Warning: NaNs could be mishandled
        
    diff_df['Overall_Tradeoff'] = np.sqrt(diff_df['Overall_Tradeoff'])
    return(diff_df)

def calc_mean_diff_with_SEM(fitness_df,conds):
# Compute mean fitness difference:
    diff_df = pd.DataFrame()
    diff_df.index = fitness_df.index
    diff_df['Source_Env'] = fitness_df['Source_Env']
    #conds = ['Glu','Gal','Lac','H2O2','NaCl']
    diff_cols = []
    
    for combination in itertools.combinations(conds, 2):
        diff_cols.append(f'{combination[0]}/{combination[1]}_diff')
        diff_df[f'{combination[0]}/{combination[1]}_diff'] = np.abs(fitness_df[f'{combination[0]}_fitness']-fitness_df[f'{combination[1]}_fitness'])
    
    diff_df['Mean_Diff'] = np.nan
    diff_df['Mean_Diff'] = diff_df[diff_cols].mean(axis=1) # Mean fitness difference across all pairs of conds, NaNs ignored
    
    # Define a function to compute SEM for a row
    def compute_row_sem(row):
        non_nan_values = row.dropna()
        return non_nan_values.std() / np.sqrt(len(non_nan_values))
    diff_df['SEM_Diff'] = np.nan
    # Compute SEM for each row
    sem_values = diff_df[diff_cols].apply(compute_row_sem, axis=1)
    diff_df['SEM_Diff'] = sem_values
    
    for combination in itertools.combinations(conds, 2):
        this_comb = f'{combination[0]}/{combination[1]}'
        diff_df[f'Mean_Diff_Excl_{this_comb}'] = np.nan
        diff_cols_excl = [x for x in diff_cols if x != f'{this_comb}_diff'] # Exclude only the relevant pair of conds from this calculation
        diff_df[f'Mean_Diff_Excl_{this_comb}'] = diff_df[diff_cols_excl].mean(axis=1)
        
    for combination in itertools.combinations(conds, 2):
        this_comb = f'{combination[0]}/{combination[1]}'
        first_env = combination[0]
        sec_env = combination[1]
        diff_df[f'Mean_Diff_Excl_All_{this_comb}'] = np.nan
        # Exclude any pair of conds containing either component of this pair from this calculation:
        diff_cols_excl_all = [x for x in diff_cols if first_env not in x and sec_env not in x] 
        diff_df[f'Mean_Diff_Excl_All_{this_comb}'] = diff_df[diff_cols_excl_all].mean(axis=1)
    
    diff_df['Overall_Fit'] = 0
    diff_df['Overall_Tradeoff'] = 0
    for c in conds:
        diff_df['Overall_Fit'] += np.exp(fitness_df[f'{c}_fitness'])
    
    diff_df['Sum_Fit'] = diff_df['Overall_Fit'].copy()
    diff_df['Overall_Fit'] = np.log(diff_df['Overall_Fit']/len(conds)) # Warning: NaNs could be mishandled
        
    for c in conds:
        diff_df['Overall_Tradeoff'] += (np.exp(fitness_df[f'{c}_fitness'])/diff_df['Sum_Fit'] - 1/len(conds))**2 # Warning: NaNs could be mishandled
        
    diff_df['Overall_Tradeoff'] = np.sqrt(diff_df['Overall_Tradeoff'])
    return(diff_df)


def calc_mean_angle_diff(fitness_df,conds):
    import math
    diff_df = pd.DataFrame()
    diff_df.index = fitness_df.index
    diff_df['Source_Env'] = fitness_df['Source_Env']
    #conds = ['Glu','Gal','Lac','H2O2','NaCl']
    diff_cols = []
    diff_sine_cols = []
    arc_cols = []
    lin_cols = []
    r_cols = []
    
    for combination in itertools.combinations(conds, 2):
        diff_cols.append(f'{combination[0]}/{combination[1]}_angle_diff')
        diff_sine_cols.append(f'{combination[0]}/{combination[1]}_angle_sine_diff')
        arc_cols.append(f'{combination[0]}/{combination[1]}_arc_diff')
        lin_cols.append(f'{combination[0]}/{combination[1]}_lin_diff')
        r_cols.append(f'{combination[0]}/{combination[1]}_r')
        diff_df[f'{combination[0]}/{combination[1]}_r'] = np.sqrt(fitness_df[f'{combination[0]}_fitness']**2 + 
                                                                        fitness_df[f'{combination[1]}_fitness']**2)
        for k in fitness_df.index:
            x = fitness_df.loc[k,f'{combination[0]}_fitness']
            y = fitness_df.loc[k,f'{combination[1]}_fitness']
            r = np.sqrt(x**2 + y**2)
            angle = math.degrees(math.atan2(y,x))
            if angle>0:
                if angle<135:
                    diff_df.loc[k,f'{combination[0]}/{combination[1]}_angle_diff'] = math.radians(np.abs(angle-45))
                    diff_df.loc[k,f'{combination[0]}/{combination[1]}_angle_sine_diff'] = np.sin(math.radians(np.abs(angle-45)))
                else:
                    diff_df.loc[k,f'{combination[0]}/{combination[1]}_angle_diff'] = math.radians(45+180-angle)
                    diff_df.loc[k,f'{combination[0]}/{combination[1]}_angle_sine_diff'] = np.sin(math.radians(45+180-angle))
            elif angle>-45:
                diff_df.loc[k,f'{combination[0]}/{combination[1]}_angle_diff'] = math.radians(45-angle)
                diff_df.loc[k,f'{combination[0]}/{combination[1]}_angle_sine_diff'] = np.sin(math.radians(45-angle))
            else:
                diff_df.loc[k,f'{combination[0]}/{combination[1]}_angle_diff'] = math.radians(np.abs(angle+135))
                diff_df.loc[k,f'{combination[0]}/{combination[1]}_angle_sine_diff'] = np.sin(math.radians(np.abs(angle+135)))
            
    diff_df['Mean_Angle_Diff'] = np.nan
    diff_df['Mean_Angle_Diff'] = diff_df[diff_cols].mean(axis=1)
    
    diff_df['Mean_Sine_Angle_Diff'] = np.nan
    diff_df['Mean_Sine_Angle_Diff'] = diff_df[diff_sine_cols].mean(axis=1)
    
    diff_df['Mean_Origin_Distance'] = np.nan
    diff_df['Mean_Origin_Distance'] = diff_df[r_cols].mean(axis=1)
    
    return(diff_df)



def calc_nonadd(fitness_df,pair_conds,pair_conds_mean):
    # Calculate nonadditivity for each fluct env and mean nonadditivity overall
    # pair_conds lists all pairs for finding nonadditivity; pair_conds_mean lists pairs for averaging nonaddivity (i.e. leave out Lac/H from the second list)
    nonadd_df = pd.DataFrame()
    nonadd_df.index = fitness_df.index
    for c in range(len(pair_conds)):
        nonadd_df[f'{pair_conds[c][2]} Predicted Fluctuating Fitness'] = np.nan
        nonadd_df[f'{pair_conds[c][2]} Predicted Fluctuating Fitness'] = fitness_df[f'{pair_conds[c][0]}_fitness']*(.5)+fitness_df[f'{pair_conds[c][1]}_fitness']*(.5)
        nonadd_df[f'{pair_conds[c][2]} Nonadditivity'] = np.abs(nonadd_df[f'{pair_conds[c][2]} Predicted Fluctuating Fitness'] - fitness_df[f'{pair_conds[c][2]}_fitness'])
            
    nonadd_df['Mean Nonadditivity'] = 0
    for BC in fitness_df.index:
        not_nan=0
        for c in range(len(pair_conds_mean)):
            if not np.isnan(nonadd_df.loc[BC,f'{pair_conds_mean[c][2]} Nonadditivity']):
                nonadd_df.loc[BC,'Mean Nonadditivity'] += nonadd_df.loc[BC,f'{pair_conds_mean[c][2]} Nonadditivity']
                not_nan += 1
        
        nonadd_df.loc[BC,'Mean Nonadditivity'] = nonadd_df.loc[BC,'Mean Nonadditivity']/not_nan
    return(nonadd_df)


def calc_nonadd_with_SEM(fitness_df,pair_conds,pair_conds_mean):
    # Calculate nonadditivity for each fluct env and mean nonadditivity overall
    # pair_conds lists all pairs for finding nonadditivity; pair_conds_mean lists pairs for averaging nonaddivity (i.e. leave out Lac/H from the second list)
    nonadd_df = pd.DataFrame()
    nonadd_df.index = fitness_df.index
    for c in range(len(pair_conds)):
        nonadd_df[f'{pair_conds[c][2]} Predicted Fluctuating Fitness'] = np.nan
        nonadd_df[f'{pair_conds[c][2]} Predicted Fluctuating Fitness'] = fitness_df[f'{pair_conds[c][0]}_fitness']*(.5)+fitness_df[f'{pair_conds[c][1]}_fitness']*(.5)
        nonadd_df[f'{pair_conds[c][2]} Nonadditivity'] = np.abs(nonadd_df[f'{pair_conds[c][2]} Predicted Fluctuating Fitness'] - fitness_df[f'{pair_conds[c][2]}_fitness'])
            
    nonadd_df['Mean Nonadditivity'] = 0
    nonadd_df['SEM Nonadditivity'] = np.nan
    for BC in fitness_df.index:
        not_nan=0
        nonadd_vals=[]
        for c in range(len(pair_conds_mean)):
            if not np.isnan(nonadd_df.loc[BC,f'{pair_conds_mean[c][2]} Nonadditivity']):
                nonadd_df.loc[BC,'Mean Nonadditivity'] += nonadd_df.loc[BC,f'{pair_conds_mean[c][2]} Nonadditivity']
                nonadd_vals.append(nonadd_df.loc[BC,f'{pair_conds_mean[c][2]} Nonadditivity'])
                not_nan += 1
        
        nonadd_df.loc[BC,'Mean Nonadditivity'] = nonadd_df.loc[BC,'Mean Nonadditivity']/not_nan
        nonadd_df.loc[BC,'SEM Nonadditivity'] = np.std(nonadd_vals)/np.sqrt(len(nonadd_vals))
        
    return(nonadd_df)




def calc_memory(fitness_df,pair_conds,pair_conds_mean):
    # Calculate memory for each fluct env and mean memory overall
    # pair_conds lists all pairs for finding memory; pair_conds_mean lists pairs for averaging memory (i.e. leave out Lac/H from the second list)
    mem_df = pd.DataFrame()
    mem_df.index = fitness_df.index
    for c in range(len(pair_conds)):
        # component 1:
        mem_df[f'{pair_conds[c][0]}_{pair_conds[c][2]} Memory'] = np.abs(fitness_df[f'{pair_conds[c][0]}_fitness']-fitness_df[f'{pair_conds[c][0]}_{pair_conds[c][2]}_fitness'])
        # component 2:
        mem_df[f'{pair_conds[c][1]}_{pair_conds[c][2]} Memory'] = np.abs(fitness_df[f'{pair_conds[c][1]}_fitness']-fitness_df[f'{pair_conds[c][1]}_{pair_conds[c][2]}_fitness'])
        # both components averaged:
        mem_df[f'mean_{pair_conds[c][2]}_memory'] = (mem_df[f'{pair_conds[c][0]}_{pair_conds[c][2]} Memory'] + mem_df[f'{pair_conds[c][1]}_{pair_conds[c][2]} Memory'])/2
        
    mem_df['Mean Memory'] = 0
    for BC in fitness_df.index:
        not_nan=0
        for c in range(len(pair_conds_mean)):
            if not np.isnan(mem_df.loc[BC,f'{pair_conds_mean[c][0]}_{pair_conds_mean[c][2]} Memory']):
                if not np.isnan(mem_df.loc[BC,f'{pair_conds_mean[c][1]}_{pair_conds_mean[c][2]} Memory']):
                    mem_df.loc[BC,'Mean Memory'] += mem_df.loc[BC,f'{pair_conds_mean[c][0]}_{pair_conds_mean[c][2]} Memory']
                    mem_df.loc[BC,'Mean Memory'] += mem_df.loc[BC,f'{pair_conds_mean[c][1]}_{pair_conds_mean[c][2]} Memory']
                    not_nan += 1
            
        mem_df.loc[BC,'Mean Memory'] = mem_df.loc[BC,'Mean Memory']/(2*not_nan)
    
    return (mem_df)


def calc_memory_with_SEM(fitness_df,pair_conds,pair_conds_mean):
    # Calculate memory for each fluct env and mean memory overall
    # pair_conds lists all pairs for finding memory; pair_conds_mean lists pairs for averaging memory (i.e. leave out Lac/H from the second list)
    mem_df = pd.DataFrame()
    mem_df.index = fitness_df.index
    for c in range(len(pair_conds)):
        # component 1:
        mem_df[f'{pair_conds[c][0]}_{pair_conds[c][2]} Memory'] = np.abs(fitness_df[f'{pair_conds[c][0]}_fitness']-fitness_df[f'{pair_conds[c][0]}_{pair_conds[c][2]}_fitness'])
        # component 2:
        mem_df[f'{pair_conds[c][1]}_{pair_conds[c][2]} Memory'] = np.abs(fitness_df[f'{pair_conds[c][1]}_fitness']-fitness_df[f'{pair_conds[c][1]}_{pair_conds[c][2]}_fitness'])
        # both components averaged:
        mem_df[f'mean_{pair_conds[c][2]}_memory'] = (mem_df[f'{pair_conds[c][0]}_{pair_conds[c][2]} Memory'] + mem_df[f'{pair_conds[c][1]}_{pair_conds[c][2]} Memory'])/2
        
    mem_df['Mean Memory'] = 0
    mem_df['SEM Memory'] = np.nan
    for BC in fitness_df.index:
        not_nan=0
        mem_vals=[]
        for c in range(len(pair_conds_mean)):
            if not np.isnan(mem_df.loc[BC,f'{pair_conds_mean[c][0]}_{pair_conds_mean[c][2]} Memory']):
                if not np.isnan(mem_df.loc[BC,f'{pair_conds_mean[c][1]}_{pair_conds_mean[c][2]} Memory']):
                    mem_df.loc[BC,'Mean Memory'] += mem_df.loc[BC,f'{pair_conds_mean[c][0]}_{pair_conds_mean[c][2]} Memory']
                    mem_df.loc[BC,'Mean Memory'] += mem_df.loc[BC,f'{pair_conds_mean[c][1]}_{pair_conds_mean[c][2]} Memory']
                    mem_vals.append(mem_df.loc[BC,f'{pair_conds_mean[c][0]}_{pair_conds_mean[c][2]} Memory'])
                    mem_vals.append(mem_df.loc[BC,f'{pair_conds_mean[c][1]}_{pair_conds_mean[c][2]} Memory'])
                    not_nan += 1
            
        mem_df.loc[BC,'Mean Memory'] = mem_df.loc[BC,'Mean Memory']/(2*not_nan)
        mem_df.loc[BC,'SEM Memory'] = np.std(mem_vals)/np.sqrt(len(mem_vals))
    
    return (mem_df)

# %%
# Import fitness dfs
formatted_data_path = 'formatted_data/'
non_neutralAB_withT0 = pd.read_csv(f'{formatted_data_path}non_neutralAB_fitness_withT0.csv',index_col=0)
non_neutralAB_withoutT0 = pd.read_csv(f'{formatted_data_path}non_neutralAB_fitness_withoutT0.csv',index_col=0)
non_neutralAB_withT0_rm_high_pool_frac = pd.read_csv(f'{formatted_data_path}non_neutralAB_fitness_withT0_exceptions.csv',index_col=0)
non_neutralAB_withoutT0_rm_high_pool_frac = pd.read_csv(f'{formatted_data_path}non_neutralAB_fitness_withoutT0_exceptions.csv',index_col=0)
adaptive_withT0 = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_fitness_withT0.csv',index_col=0)
adaptive_withT0_high = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_fitness_withT0_0.1_threshold.csv',index_col=0)
adaptive_withT0_rm_high_pool_frac = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_fitness_withT0_exceptions.csv',index_col=0)
adaptive_withoutT0 = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_fitness_withoutT0.csv',index_col=0)
adaptive_withoutT0_high = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_fitness_withoutT0_0.1_threshold.csv',index_col=0)
adaptive_withoutT0_rm_high_pool_frac = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_fitness_withoutT0_exceptions.csv',index_col=0)

adaptive_withT0_rm_high_pool_frac_high_thr = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_fitness_withT0_0.1_threshold_exceptions.csv',index_col=0)
adaptive_withoutT0_rm_high_pool_frac_high_thr = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_fitness_withoutT0_0.1_threshold_exceptions.csv',index_col=0)

# Choose adaptive threshold, whether to include timepoint T0 in fitness inferences, whether to drop Lac-H2O2 and high-batch effect conds from averages:
thr = 'normal'
choose = 'not_rm_high_pool_frac'
drop_LacH = True
drop_batch_fx = False

if thr == 'normal':
    if choose == 'T0':
        fitness_df = adaptive_withT0
    elif choose == 'not':
        fitness_df = adaptive_withoutT0
    elif choose == 'T0_rm_high_pool_frac':
        fitness_df = adaptive_withT0_rm_high_pool_frac
    elif choose == 'not_rm_high_pool_frac':
        fitness_df = adaptive_withoutT0_rm_high_pool_frac
    elif choose == 'all_T0':
        fitness_df = non_neutralAB_withT0
    elif choose == 'all_not':
        fitness_df = non_neutralAB_withoutT0
    elif choose == 'all_T0_rm_high_pool_frac':
        fitness_df = non_neutralAB_withT0_rm_high_pool_frac
    elif choose == 'all_not_rm_high_pool_frac':
        fitness_df = non_neutralAB_withoutT0_rm_high_pool_frac
    
    else:
        print('ERROR: CHOOSE "T0" or "not"')
elif thr == 'high':
    if choose == 'T0':
        fitness_df = adaptive_withT0_high
    elif choose == 'not':
        fitness_df = adaptive_withoutT0_high
            
    elif choose == 'T0_rm_high_pool_frac':
        fitness_df = adaptive_withT0_rm_high_pool_frac_high_thr
    elif choose == 'not_rm_high_pool_frac':
        fitness_df = adaptive_withoutT0_rm_high_pool_frac_high_thr
    else:
        print('ERROR: CHOOSE "T0" or "not"')

#fitness_df = adaptive_withT0

# Calculate mean fitness difference across static envs:

static_conds = ['Glu','Gal','Lac','H2O2','NaCl']

if drop_batch_fx:
    fluct_conds = ['Glu/Gal','Glu/Lac','Glu/NaCl','H2O2/NaCl','Gal/H2O2',
                   'Gal/NaCl','Lac/NaCl']
    all_conds = ['Glu','Gal','Lac','H2O2','NaCl','Glu/Gal','Glu/Lac','Gal/Lac',
            'Glu/H2O2','Glu/NaCl','H2O2/NaCl','Gal/H2O2','Gal/NaCl','Lac/NaCl']
    
elif drop_LacH:
    fluct_conds = ['Glu/Gal','Glu/Lac','Gal/Lac','Glu/H2O2','Glu/NaCl',
               'H2O2/NaCl','Gal/H2O2','Gal/NaCl','Lac/NaCl']
    all_conds = ['Glu','Gal','Lac','H2O2','NaCl','Glu/Gal','Glu/Lac','Gal/Lac',
            'Glu/H2O2','Glu/NaCl','H2O2/NaCl','Gal/H2O2','Gal/NaCl','Lac/NaCl']
    
else:
    fluct_conds = ['Glu/Gal','Glu/Lac','Gal/Lac','Glu/H2O2','Glu/NaCl',
               'H2O2/NaCl','Gal/H2O2','Gal/NaCl','Lac/NaCl','Lac/H2O2']
    all_conds = ['Glu','Gal','Lac','H2O2','NaCl','Glu/Gal','Glu/Lac','Gal/Lac',
            'Glu/H2O2','Glu/NaCl','H2O2/NaCl','Gal/H2O2','Gal/NaCl','Lac/NaCl','Lac/H2O2']
    
exp_conds = ['Glu','Gal','Lac','H2O2','NaCl','Glu/Gal','Glu/Lac',
             'Glu/H2O2','Glu/NaCl','H2O2/NaCl']

static_diff_df = calc_mean_diff_with_SEM(fitness_df,static_conds)
static_angle_diff_df = calc_mean_angle_diff(fitness_df,static_conds)
# Merge static fit diff columns into ftiness_df:
for combination in itertools.combinations(static_conds, 2):
    if drop_LacH:
        if combination[0] != 'Lac':
            fitness_df = fitness_df.join(static_diff_df[f'{combination[0]}/{combination[1]}_diff'])
        elif combination[1] != 'H2O2':
            fitness_df = fitness_df.join(static_diff_df[f'{combination[0]}/{combination[1]}_diff'])
    else:
        fitness_df = fitness_df.join(static_diff_df[f'{combination[0]}/{combination[1]}_diff'])
        
fitness_df = fitness_df.join(static_diff_df['Mean_Diff'].rename('Mean_Diff_Static'))
fitness_df = fitness_df.join(static_diff_df['SEM_Diff'].rename('SEM_Diff_Static'))
fitness_df = fitness_df.join(static_angle_diff_df['Mean_Angle_Diff'].rename('Mean_Angle_Diff_Static'))
fitness_df = fitness_df.join(static_angle_diff_df['Mean_Sine_Angle_Diff'].rename('Mean_Sine_Angle_Diff_Static'))
fitness_df = fitness_df.join(static_angle_diff_df['Mean_Origin_Distance'].rename('Mean_Origin_Distance_Static'))
fitness_df = fitness_df.join(static_diff_df['Overall_Fit'].rename('Overall_Fit_Static'))
fitness_df = fitness_df.join(static_diff_df['Overall_Tradeoff'].rename('Overall_Tradeoff_Static'))

# Calculate mean fitness difference across all envs:

all_diff_df = calc_mean_diff(fitness_df,all_conds)
all_angle_diff_df = calc_mean_angle_diff(fitness_df,all_conds)
# Merge all fit diff columns into ftiness_df:
fitness_df = fitness_df.join(all_diff_df['Mean_Diff'].rename('Mean_Diff_All'))
fitness_df = fitness_df.join(all_angle_diff_df['Mean_Angle_Diff'].rename('Mean_Angle_Diff_All'))
fitness_df = fitness_df.join(all_angle_diff_df['Mean_Sine_Angle_Diff'].rename('Mean_Sine_Angle_Diff_All'))
fitness_df = fitness_df.join(all_angle_diff_df['Mean_Origin_Distance'].rename('Mean_Origin_Distance_All'))
fitness_df = fitness_df.join(all_diff_df['Overall_Fit'].rename('Overall_Fit_All'))
fitness_df = fitness_df.join(all_diff_df['Overall_Tradeoff'].rename('Overall_Tradeoff_All'))

# Calculate memory:
Gal_conds = ['Glu','Gal','Glu/Gal']
N_conds = ['Glu','NaCl','Glu/NaCl']
L_conds = ['Glu','Lac','Glu/Lac']
H_conds = ['Glu','H2O2','Glu/H2O2']
HN_conds = ['H2O2','NaCl','H2O2/NaCl']
GalL_conds = ['Gal','Lac','Gal/Lac']
GalH_conds = ['Gal','H2O2','Gal/H2O2']
GalN_conds = ['Gal','NaCl','Gal/NaCl']
LN_conds = ['Lac','NaCl','Lac/NaCl']
LH_conds = ['Lac','H2O2','Lac/H2O2']
pair_conds = [Gal_conds,L_conds,GalL_conds,H_conds,N_conds,HN_conds,GalH_conds,GalN_conds,LN_conds,LH_conds]
if drop_batch_fx:
    pair_conds_mean = [Gal_conds,L_conds,N_conds,HN_conds,GalH_conds,GalN_conds,LN_conds]
elif drop_LacH:
    pair_conds_mean = [Gal_conds,L_conds,GalL_conds,H_conds,N_conds,HN_conds,GalH_conds,GalN_conds,LN_conds]
else:
    pair_conds_mean = [Gal_conds,L_conds,GalL_conds,H_conds,N_conds,HN_conds,GalH_conds,GalN_conds,LN_conds,LH_conds]
    
nonadd_df = calc_nonadd_with_SEM(fitness_df,pair_conds,pair_conds_mean)

fitness_df = fitness_df.join(nonadd_df['Mean Nonadditivity'].rename('Mean_Nonadditivity'))
fitness_df = fitness_df.join(nonadd_df['SEM Nonadditivity'].rename('SEM_Nonadditivity'))

mem_df = calc_memory_with_SEM(fitness_df,pair_conds,pair_conds_mean)
fitness_df = fitness_df.join(mem_df['Mean Memory'].rename('Mean_Memory'))
fitness_df = fitness_df.join(mem_df['SEM Memory'].rename('SEM_Memory'))

if thr == 'normal':
    
    
    if drop_batch_fx:
        if choose == 'T0':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_drop_batch_fx.csv')
        elif choose == 'not':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_drop_batch_fx.csv')
        elif choose == 'H_hybrid':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_H_hybrid_overall_stats_drop_batch_fx.csv')
        elif choose == 'T0_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_rm_high_pool_frac_drop_batch_fx.csv')
        elif choose == 'not_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac_drop_batch_fx.csv')
        elif choose == 'all_T0':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withT0_overall_stats_drop_batch_fx.csv')
        elif choose == 'all_not':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withoutT0_overall_stats_drop_batch_fx.csv')
        elif choose == 'all_T0_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withT0_overall_stats_rm_high_pool_frac_drop_batch_fx.csv')
        elif choose == 'all_not_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac_drop_batch_fx.csv')
            
    elif drop_LacH:
        if choose == 'T0':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_drop_LacH.csv')
        elif choose == 'not':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_drop_LacH.csv')
        elif choose == 'H_hybrid':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_H_hybrid_overall_stats_drop_LacH.csv')
        elif choose == 'T0_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_rm_high_pool_frac_drop_LacH.csv')
        elif choose == 'not_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac_drop_LacH.csv')
        elif choose == 'all_T0':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withT0_overall_stats_drop_LacH.csv')
        elif choose == 'all_not':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withoutT0_overall_stats_drop_LacH.csv')
        elif choose == 'all_T0_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withT0_overall_stats_rm_high_pool_frac_drop_LacH.csv')
        elif choose == 'all_not_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac_drop_LacH.csv')
            
    else:
        if choose == 'T0':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats.csv')
        elif choose == 'not':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats.csv')
        elif choose == 'H_hybrid':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_H_hybrid_overall_stats.csv')
        elif choose == 'T0_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_rm_high_pool_frac.csv')
        elif choose == 'not_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac.csv')
        elif choose == 'all_T0':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withT0_overall_stats.csv')
        elif choose == 'all_not':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withoutT0_overall_stats.csv')
        elif choose == 'all_T0_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withT0_overall_stats_rm_high_pool_frac.csv')
        elif choose == 'all_not_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/all_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac.csv')
            
elif thr == 'high':
    if drop_LacH:
        if choose == 'T0':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_drop_LacH_0.1_threshold.csv')
        elif choose == 'not':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_drop_LacH_0.1_threshold.csv')
        elif choose == 'H_hybrid':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_H_hybrid_overall_stats_drop_LacH_0.1_threshold.csv')
        elif choose == 'T0_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_rm_high_pool_frac_drop_LacH_0.1_threshold.csv')
        elif choose == 'not_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac_drop_LacH_0.1_threshold.csv')
    else:
        if choose == 'T0':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_0.1_threshold.csv')
        elif choose == 'not':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_0.1_threshold.csv')
        elif choose == 'H_hybrid':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_H_hybrid_overall_stats_0.1_threshold.csv')
        elif choose == 'T0_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withT0_overall_stats_rm_high_pool_frac_0.1_threshold.csv')
        elif choose == 'not_rm_high_pool_frac':
            fitness_df.to_csv('summary_data/adaptive_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac_0.1_threshold.csv')


#%% Include all static fitness stats:

all_static_df = static_diff_df.merge(mem_df, left_index=True, right_index=True)
all_static_df = all_static_df.merge(static_angle_diff_df, left_index=True, right_index=True)
all_static_df = all_static_df.drop('Source_Env_y', axis=1)
all_static_df = all_static_df.rename(columns={'Source_Env_x': 'Source_Env'})
all_static_df = all_static_df.join(nonadd_df['Mean Nonadditivity'].rename('Mean_Nonadditivity'))
all_static_df = all_static_df.join(mem_df['Mean Memory'].rename('Mean_Memory'))

for pair in fluct_conds:
    all_static_df = all_static_df.join(nonadd_df[f'{pair} Nonadditivity'].rename(f'{pair}_nonadditivity'))

if thr == 'normal':
    if drop_LacH:
        if choose == 'T0':    
            all_static_df.to_csv('summary_data/all_static_withT0_overall_stats_drop_LacH.csv')
        elif choose == 'not':
            all_static_df.to_csv('summary_data/all_static_withoutT0_overall_stats_drop_LacH.csv')
        elif choose == 'T0_rm_high_pool_frac':
            all_static_df.to_csv('summary_data/all_static_withT0_overall_stats_rm_high_pool_frac_drop_LacH.csv')
        elif choose == 'not_rm_high_pool_frac':
            all_static_df.to_csv('summary_data/all_static_withoutT0_overall_stats_rm_high_pool_frac_drop_LacH.csv')
    else:
        if choose == 'T0':    
            all_static_df.to_csv('summary_data/all_static_withT0_overall_stats.csv')
        elif choose == 'not':
            all_static_df.to_csv('summary_data/all_static_withoutT0_overall_stats.csv')
        elif choose == 'T0_rm_high_pool_frac':
            all_static_df.to_csv('summary_data/all_static_withT0_overall_stats_rm_high_pool_frac.csv')
        elif choose == 'not_rm_high_pool_frac':
            all_static_df.to_csv('summary_data/all_static_withoutT0_overall_stats_rm_high_pool_frac.csv')
elif thr == 'high':
    if drop_LacH:
        if choose == 'T0':    
            all_static_df.to_csv('summary_data/all_static_withT0_overall_stats_drop_LacH_0.1_threshold.csv')
        elif choose == 'not':
            all_static_df.to_csv('summary_data/all_static_withoutT0_overall_stats_drop_LacH_0.1_threshold.csv')
    else:
        if choose == 'T0':    
            all_static_df.to_csv('summary_data/all_static_withT0_overall_stats_0.1_threshold.csv')
        elif choose == 'not':
            all_static_df.to_csv('summary_data/all_static_withoutT0_overall_stats_0.1_threshold.csv')
