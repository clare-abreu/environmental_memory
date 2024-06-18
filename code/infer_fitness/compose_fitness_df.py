#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:51:19 2023

@author: clare
"""

import pandas as pd
import numpy as np


# %%
# Import fitness dfs
data_path = '../../data/'
BClist = pd.read_csv(f'{data_path}BClist.csv',index_col=0)
formatted_data_path = 'summary_data/'
adaptive_withoutT0_rm_high_pool_frac_dropLacH = pd.read_csv(f'{formatted_data_path}adaptive_non_neutralAB_withoutT0_overall_stats_rm_high_pool_frac_drop_LacH.csv',index_col=0)
# Bootstrapped SEMs:
formatted_data_path = 'formatted_data/'
bootstrap_withoutT0_exc = pd.read_csv(f'{formatted_data_path}bootstrapped_non_neutralAB_fitness_withoutT0_with_reps_exceptions.csv',index_col=0)

fitness_df = adaptive_withoutT0_rm_high_pool_frac_dropLacH
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

# Drop some columns:
fitness_df_to_save = fitness_df.copy()
strings = ['_naive_SEM']
# Filter columns based on the condition
for string in strings:
    columns_to_drop = [col for col in fitness_df.columns if col.endswith(string)]
# Drop the columns
    fitness_df_to_save = fitness_df_to_save.drop(columns=columns_to_drop)

#%% Save fitness dataframe:
    
fitness_df_to_save.to_csv('../../data/fitness_df.csv')

