#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:51:19 2023

@author: clare
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from scipy.stats import linregress
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
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

sns.set_color_codes()
sns.set_style('white')
sns.set_style('ticks')

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%% Color dictionary

color_dict = {}
color_dict['DPY256'] = 'black'
color_dict['Venkataram'] = 'darkgray'
color_dict['Glu'] = 'white'
color_dict['H2O2'] = '#055deb'#'blue'#'#fa0505'
color_dict['Glu/H2O2'] = '#db8686'
color_dict['H2O2/NaCl'] = '#f59c05'
color_dict['NaCl'] = 'yellow'#'#bdba0f'
color_dict['Glu/NaCl'] = '#918f41'
color_dict['Lac'] = 'purple'#'#055deb'
color_dict['Glu/Lac'] = '#5d7db0'
color_dict['Gal'] = 'pink'#'#05ab58'
color_dict['Glu/Gal'] = '#539172'

color_dict_gray = {}
color_dict_gray['DPY256'] = 'black'
color_dict_gray['Venkataram'] = 'darkgray'
color_dict_gray['Glu'] = 'white'
color_dict_gray['H2O2'] = 'white'#'blue'#'#fa0505'
color_dict_gray['Glu/H2O2'] = '#db8686'
color_dict_gray['H2O2/NaCl'] = '#f59c05'
color_dict_gray['NaCl'] = 'gray'#'#bdba0f'
color_dict_gray['Glu/NaCl'] = '#918f41'
color_dict_gray['Lac'] = 'gray'#'#055deb'
color_dict_gray['Glu/Lac'] = '#5d7db0'
color_dict_gray['Gal'] = 'gray'#'#05ab58'
color_dict_gray['Glu/Gal'] = '#539172'

# %%
# Import fitness dfs
data_path = '../../data/'
BClist = pd.read_csv(f'{data_path}BClist.csv',index_col=0)
formatted_data_path = 'summary_data/'
#adaptive_withT0 = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_withT0_overall_stats.csv',index_col=0)
#adaptive_withT0_rm_high_pool_frac = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_withT0_overall_stats_rm_high_pool_frac.csv',index_col=0)
#adaptive_all_withT0 = pd.read_csv(f'{formatted_data_path}all_static_withT0_overall_stats.csv',index_col=0)
#adaptive_withoutT0 = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_withoutT0_overall_stats.csv',index_col=0)
adaptive_withoutT0_rm_high_pool_frac = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac.csv',index_col=0)
adaptive_withoutT0_rm_high_pool_frac_dropLacH = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac_drop_LacH.csv',index_col=0)
# Bootstrapped SEMs:
formatted_data_path = 'formatted_data/'
bootstrap_withoutT0_exc = pd.read_csv(f'{formatted_data_path}bootstrapped_non_neutralAB_fitness_withoutT0_with_reps_exceptions.csv',index_col=0)
bootstrap_withT0_exc = pd.read_csv(f'{formatted_data_path}bootstrapped_non_neutralAB_fitness_withT0_with_reps_exceptions.csv',index_col=0)

choose = 'not_rm_high_pool_frac'
if choose == 'T0':
    fitness_df = adaptive_withT0
elif choose == 'not':
    fitness_df = adaptive_withoutT0
elif choose == 'T0_rm_high_pool_frac':
    fitness_df = adaptive_withT0_rm_high_pool_frac
elif choose == 'not_rm_high_pool_frac':
    fitness_df = adaptive_withoutT0_rm_high_pool_frac_dropLacH
else:
    print('ERROR: CHOOSE "T0" or "not"')

# High-variance mutants:
df_high_rep_variation_mutants = pd.read_csv('high_rep_variation_mutants.csv',index_col=0)
high_rep_variation_mutants = df_high_rep_variation_mutants.index.tolist()
fitness_df_dropped = fitness_df.drop(high_rep_variation_mutants)

# COMMENT OUT IF NOT DROPPING HIGH VARIANCE MUTANTS:
#fitness_df = fitness_df_dropped

if choose == 'T0_rm_high_pool_frac':
    bootstrap_df = bootstrap_withT0_exc
elif choose == 'not_rm_high_pool_frac':
    bootstrap_df = bootstrap_withoutT0_exc

for cond in ['Glu','Gal','Lac','H2O2','NaCl']:
    fitness_df[f'{cond}_SEM_bootstrap'] = np.nan
    for BC in fitness_df.index.tolist():
        fitness_df.loc[BC,f'{cond}_SEM_bootstrap'] = bootstrap_df.loc[BC,f'{cond}_SEM_bootstrap']
        fitness_df.loc[BC,f'{cond}_SEM_across_reps'] = bootstrap_df.loc[BC,f'{cond}_SEM_across_reps']
        
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

for cond in pair_conds:
    fluct_cond = cond[2]
    comp1 = cond[0]
    comp2 = cond[1]
    fitness_df[f'{fluct_cond}_SEM_bootstrap'] = np.nan
    fitness_df[f'{comp1}_{fluct_cond}_SEM_bootstrap'] = np.nan
    fitness_df[f'{comp2}_{fluct_cond}_SEM_bootstrap'] = np.nan
    for BC in fitness_df.index.tolist():
        fitness_df.loc[BC,f'{comp1}_{fluct_cond}_SEM_bootstrap'] = bootstrap_df.loc[BC,f'{comp1}_{fluct_cond}_SEM_bootstrap']
        fitness_df.loc[BC,f'{comp2}_{fluct_cond}_SEM_bootstrap'] = bootstrap_df.loc[BC,f'{comp2}_{fluct_cond}_SEM_bootstrap']
        fitness_df.loc[BC,f'{fluct_cond}_SEM_bootstrap'] = 0.5*np.sqrt(bootstrap_df.loc[BC,f'{comp1}_{fluct_cond}_SEM_bootstrap']**2 + 
                                                                       bootstrap_df.loc[BC,f'{comp2}_{fluct_cond}_SEM_bootstrap']**2)
        
        fitness_df.loc[BC,f'{comp1}_{fluct_cond}_SEM_across_reps'] = bootstrap_df.loc[BC,f'{comp1}_{fluct_cond}_SEM_across_reps']
        fitness_df.loc[BC,f'{comp2}_{fluct_cond}_SEM_across_reps'] = bootstrap_df.loc[BC,f'{comp2}_{fluct_cond}_SEM_across_reps']
        fitness_df.loc[BC,f'{fluct_cond}_SEM_across_reps'] = 0.5*np.sqrt(bootstrap_df.loc[BC,f'{comp1}_{fluct_cond}_SEM_across_reps']**2 + 
                                                                       bootstrap_df.loc[BC,f'{comp2}_{fluct_cond}_SEM_across_reps']**2)

# SAVE FITNESS DATAFRAME:
fitness_df_to_save = fitness_df.copy()
strings = ['_naive_SEM','_Static']
# Filter columns based on the condition
for string in strings:
    columns_to_drop = [col for col in fitness_df.columns if col.endswith(string)]
# Drop the columns
    fitness_df_to_save = fitness_df_to_save.drop(columns=columns_to_drop)

#%% Save fitness dataframe:
    
fitness_df_to_save.to_csv('../../data/fitness_df.csv')

