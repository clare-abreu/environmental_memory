#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:36:42 2023

@author: clare
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from scipy.stats import linregress
from scipy.spatial import distance
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
def add_src_envs(df,BClist):
    # Add source environments
    df['Source_Env'] = ""
    for k in df.index:
        df.loc[k,'Source_Env'] = BClist.loc[k,'Source_Env']
    return(df)


def make_non_neutralAB_df(df_static,df_fluct_comp1,df_fluct_comp2,BClist,add_exceptions=False,df_static_exceptions=None,df_fluct_comp2_exceptions=None):
    # Make df of fitness of non-neutral and non-Alice/Bob barcodes
    # Need to generalize to other types of SEM than 'Naive_std_err'
    # If add_exceptions=True, this means that we are excluding timepoints from fitness estimates where the bc pool grows to a total freq > 0.5

    non_neutral_nonAB_BCs = BClist[(BClist['Source_Env']!='Neutral') & (BClist['Source_Env']!='Ancestor') & (BClist['Source_Env']!='Alice') &
                                   (BClist['Source_Env']!='Bob')].index
    non_neutralAB_df = pd.DataFrame()
    non_neutralAB_df.index = non_neutral_nonAB_BCs
    non_neutralAB_df['Source_Env'] = BClist.loc[non_neutral_nonAB_BCs]['Source_Env']

    if add_exceptions:
        static_conds = ['Glu','Gal','Lac','NaCl']
        static_conds_exceptions = ['H2O2']

        fluct_conds = ['Glu/Gal','Glu/Lac','Glu/NaCl','H2O2/NaCl','Gal/H2O2','Gal/NaCl','Lac/NaCl']
        fluct_conds_exceptions = ['Glu/H2O2','Gal/Lac','Lac/H2O2']
        
    else:
        static_conds = ['Glu','Gal','Lac','H2O2','NaCl']
        fluct_conds = ['Glu/Gal','Glu/Lac','Glu/H2O2','Glu/NaCl','H2O2/NaCl','Gal/Lac','Gal/H2O2','Gal/NaCl','Lac/H2O2','Lac/NaCl']
        
        
    static_dict = {'Glu':'Glu','Gal':'Gal','Lac':'L','H2O2':'high_H_H','NaCl':'N'}
    fluct_dict = {'Glu/Gal':'Glu-Gal','Glu/Lac':'Glu-L','Glu/H2O2':'high_H_Glu-H',
                  'Glu/NaCl':'Glu-N','H2O2/NaCl':'high_H_H-N','Gal/Lac':'Gal-L',
                  'Gal/H2O2':'high_H_Gal-H','Gal/NaCl':'Gal-N','Lac/H2O2':'high_H_L-H',
                  'Lac/NaCl':'L-N'}
    fluct_comp_dict = {'Glu/Gal':['Glu','Gal'],'Glu/Lac':['Glu','Lac'],'Glu/H2O2':['Glu','H2O2'],
                       'Glu/NaCl':['Glu','NaCl'],'H2O2/NaCl':['H2O2','NaCl'],'Gal/Lac':['Gal','Lac'],
                       'Gal/H2O2':['Gal','H2O2'],'Gal/NaCl':['Gal','NaCl'],
                       'Lac/H2O2':['Lac','H2O2'],'Lac/NaCl':['Lac','NaCl']}
    fluct_comp_dict_names = {'Glu/Gal':['Glu','Gal'],'Glu/Lac':['Glu','L'],'Glu/H2O2':['Glu','high_H_H'],
                       'Glu/NaCl':['Glu','N'],'H2O2/NaCl':['high_H','N'],'Gal/Lac':['Gal','L'],
                       'Gal/H2O2':['Gal','high_H_H'],'Gal/NaCl':['Gal','N'],
                       'Lac/H2O2':['L','high_H_H'],'Lac/NaCl':['L','N']}
    
    for cond in static_conds:
        cond_name = static_dict[cond]
        df_cond = df_static[df_static['condition']==cond_name]
        non_neutralAB_df[f'{cond}_fitness'] = np.nan
        non_neutralAB_df[f'{cond}_naive_SEM'] = np.nan
        for BC in non_neutral_nonAB_BCs:
            non_neutralAB_df.loc[BC,f'{cond}_fitness'] = df_cond.loc[BC,'fitness']
            non_neutralAB_df.loc[BC,f'{cond}_naive_SEM'] = df_cond.loc[BC,'Naive_std_err']
            
            
    if add_exceptions:
        for cond in static_conds_exceptions:
            cond_name = static_dict[cond]
            df_cond = df_static_exceptions[df_static_exceptions['condition']==cond_name]
            non_neutralAB_df[f'{cond}_fitness'] = np.nan
            non_neutralAB_df[f'{cond}_naive_SEM'] = np.nan
            for BC in non_neutral_nonAB_BCs:
                non_neutralAB_df.loc[BC,f'{cond}_fitness'] = df_cond.loc[BC,'fitness']
                non_neutralAB_df.loc[BC,f'{cond}_naive_SEM'] = df_cond.loc[BC,'Naive_std_err']

            
    for cond in fluct_conds:
        cond_name = fluct_dict[cond]
        fluct_comps = fluct_comp_dict[cond]
        fluct_comp_names = fluct_comp_dict_names[cond]
        for c in [0,1]:
            comp = fluct_comps[c]
            comp_name = fluct_comp_names[c]
            if c == 0:
                df_cond = df_fluct_comp1[df_fluct_comp1['condition']==cond_name]
            elif c==1:
                df_cond = df_fluct_comp2[df_fluct_comp2['condition']==cond_name]
                
            non_neutralAB_df[f'{comp}_{cond}_fitness'] = np.nan
            non_neutralAB_df[f'{comp}_{cond}_naive_SEM'] = np.nan

            for BC in non_neutral_nonAB_BCs:
                non_neutralAB_df.loc[BC,f'{comp}_{cond}_fitness'] = df_cond.loc[BC,'fitness']
                non_neutralAB_df.loc[BC,f'{comp}_{cond}_naive_SEM'] = df_cond.loc[BC,'Naive_std_err']
                
    if add_exceptions:
        for cond in fluct_conds_exceptions:
            cond_name = fluct_dict[cond]
            fluct_comps = fluct_comp_dict[cond]
            fluct_comp_names = fluct_comp_dict_names[cond]
            for c in [0,1]:
                comp = fluct_comps[c]
                comp_name = fluct_comp_names[c]
                if c == 0:
                    df_cond = df_fluct_comp1[df_fluct_comp1['condition']==cond_name]
                elif c==1:
                    df_cond = df_fluct_comp2_exceptions[df_fluct_comp2_exceptions['condition']==cond_name]
                    
                non_neutralAB_df[f'{comp}_{cond}_fitness'] = np.nan
                non_neutralAB_df[f'{comp}_{cond}_naive_SEM'] = np.nan
    
                for BC in non_neutral_nonAB_BCs:
                    non_neutralAB_df.loc[BC,f'{comp}_{cond}_fitness'] = df_cond.loc[BC,'fitness']
                    non_neutralAB_df.loc[BC,f'{comp}_{cond}_naive_SEM'] = df_cond.loc[BC,'Naive_std_err']
                
    all_fluct_conds = ['Glu/Gal','Glu/Lac','Glu/H2O2','Glu/NaCl','H2O2/NaCl','Gal/Lac','Gal/H2O2','Gal/NaCl','Lac/H2O2','Lac/NaCl']
    # Average fluct fitness:
    for cond in all_fluct_conds:
        comps = fluct_comp_dict[cond]
        non_neutralAB_df[f'{cond}_fitness'] = non_neutralAB_df[[f'{comps[0]}_{cond}_fitness', f'{comps[1]}_{cond}_fitness']].mean(axis=1, skipna=False)
        non_neutralAB_df[f'{cond}_naive_SEM'] = 0.5*np.sqrt(non_neutralAB_df[f'{comps[0]}_{cond}_naive_SEM']**2 + non_neutralAB_df[f'{comps[1]}_{cond}_naive_SEM']**2 )
    
    return(non_neutralAB_df)





def make_non_neutralAB_df_R2M_control(df_static,df_fluct_comp1,df_fluct_comp2,BClist,add_exceptions=False,df_static_exceptions=None,df_fluct_comp2_exceptions=None):
    # Make df of fitness of non-neutral and non-Alice/Bob barcodes
    # R2M (regression to mean) control data ONLY

    non_neutral_nonAB_BCs = BClist[(BClist['Source_Env']!='Neutral') & (BClist['Source_Env']!='Ancestor') & (BClist['Source_Env']!='Alice') &
                                   (BClist['Source_Env']!='Bob')].index
    non_neutralAB_df = pd.DataFrame()
    non_neutralAB_df.index = non_neutral_nonAB_BCs
    non_neutralAB_df['Source_Env'] = BClist.loc[non_neutral_nonAB_BCs]['Source_Env']

    if add_exceptions:
        #static_conds = ['Glu','Gal','Lac','NaCl']
        #static_conds_exceptions = ['H2O2']

        fluct_conds = ['Glu/Gal','Glu/Lac','Glu/NaCl','H2O2/NaCl','Gal/H2O2','Gal/NaCl','Lac/NaCl']
        fluct_conds_exceptions = ['Glu/H2O2','Gal/Lac','Lac/H2O2']
        
    else:
        #static_conds = ['Glu','Gal','Lac','H2O2','NaCl']
        fluct_conds = ['Glu/Gal','Glu/Lac','Glu/H2O2','Glu/NaCl','H2O2/NaCl','Gal/Lac','Gal/H2O2','Gal/NaCl','Lac/H2O2','Lac/NaCl']
        
        
    #static_dict = {'Glu':'Glu','Gal':'Gal','Lac':'L','H2O2':'high_H_H','NaCl':'N'}
    fluct_dict = {'Glu/Gal':'Glu-Gal','Glu/Lac':'Glu-L','Glu/H2O2':'high_H_Glu-H',
                  'Glu/NaCl':'Glu-N','H2O2/NaCl':'high_H_H-N','Gal/Lac':'Gal-L',
                  'Gal/H2O2':'high_H_Gal-H','Gal/NaCl':'Gal-N','Lac/H2O2':'high_H_L-H',
                  'Lac/NaCl':'L-N'}
    fluct_comp_dict = {'Glu/Gal':['Glu','Gal'],'Glu/Lac':['Glu','Lac'],'Glu/H2O2':['Glu','H2O2'],
                       'Glu/NaCl':['Glu','NaCl'],'H2O2/NaCl':['H2O2','NaCl'],'Gal/Lac':['Gal','Lac'],
                       'Gal/H2O2':['Gal','H2O2'],'Gal/NaCl':['Gal','NaCl'],
                       'Lac/H2O2':['Lac','H2O2'],'Lac/NaCl':['Lac','NaCl']}
    fluct_comp_dict_names = {'Glu/Gal':['Glu','Gal'],'Glu/Lac':['Glu','L'],'Glu/H2O2':['Glu','high_H_H'],
                       'Glu/NaCl':['Glu','N'],'H2O2/NaCl':['high_H','N'],'Gal/Lac':['Gal','L'],
                       'Gal/H2O2':['Gal','high_H_H'],'Gal/NaCl':['Gal','N'],
                       'Lac/H2O2':['L','high_H_H'],'Lac/NaCl':['L','N']}

            
    for cond in fluct_conds:
        cond_name = fluct_dict[cond]
        fluct_comps = fluct_comp_dict[cond]
        fluct_comp_names = fluct_comp_dict_names[cond]
        for c in [0,1]:
            comp = fluct_comps[c]
            comp_name = fluct_comp_names[c]
            if c == 0:
                df_cond = df_fluct_comp1[df_fluct_comp1['condition']==cond_name]
            elif c==1:
                df_cond = df_fluct_comp2[df_fluct_comp2['condition']==cond_name]
                
            non_neutralAB_df[f'{comp}_{cond}_fitness'] = np.nan
            #non_neutralAB_df[f'{comp}_{cond}_naive_SEM'] = np.nan

            for BC in non_neutral_nonAB_BCs:
                non_neutralAB_df.loc[BC,f'{comp}_{cond}_fitness'] = df_cond.loc[BC,'fitness']
                #non_neutralAB_df.loc[BC,f'{comp}_{cond}_naive_SEM'] = df_cond.loc[BC,'Naive_std_err']
                
    if add_exceptions:
        for cond in fluct_conds_exceptions:
            cond_name = fluct_dict[cond]
            fluct_comps = fluct_comp_dict[cond]
            fluct_comp_names = fluct_comp_dict_names[cond]
            for c in [0,1]:
                comp = fluct_comps[c]
                comp_name = fluct_comp_names[c]
                if c == 0:
                    df_cond = df_fluct_comp1[df_fluct_comp1['condition']==cond_name]
                elif c==1:
                    df_cond = df_fluct_comp2_exceptions[df_fluct_comp2_exceptions['condition']==cond_name]
                    
                non_neutralAB_df[f'{comp}_{cond}_fitness'] = np.nan
                #non_neutralAB_df[f'{comp}_{cond}_naive_SEM'] = np.nan
    
                for BC in non_neutral_nonAB_BCs:
                    non_neutralAB_df.loc[BC,f'{comp}_{cond}_fitness'] = df_cond.loc[BC,'fitness']
                    #non_neutralAB_df.loc[BC,f'{comp}_{cond}_naive_SEM'] = df_cond.loc[BC,'Naive_std_err']
                
    all_fluct_conds = ['Glu/Gal','Glu/Lac','Glu/H2O2','Glu/NaCl','H2O2/NaCl','Gal/Lac','Gal/H2O2','Gal/NaCl','Lac/H2O2','Lac/NaCl']
    # Average fluct fitness:
    for cond in all_fluct_conds:
        comps = fluct_comp_dict[cond]
        non_neutralAB_df[f'{cond}_fitness'] = non_neutralAB_df[[f'{comps[0]}_{cond}_fitness', f'{comps[1]}_{cond}_fitness']].mean(axis=1, skipna=False)
        #non_neutralAB_df[f'{cond}_naive_SEM'] = 0.5*np.sqrt(non_neutralAB_df[f'{comps[0]}_{cond}_naive_SEM']**2 + non_neutralAB_df[f'{comps[1]}_{cond}_naive_SEM']**2 )
    
    return(non_neutralAB_df)





def drop_empty_static(df):
    static_conds = ['Glu','Gal','Lac','H2O2','NaCl']
    df=df.dropna(subset=[f'{x}_fitness' for x in static_conds])
    return(df)



def sep_adaptive(df,adapt_thr):
    # Separate adaptive clones
    exp_conds = ['Glu','Gal','Lac','H2O2','NaCl','Glu/Gal',
                 'Glu/Lac','Glu/H2O2','Glu/NaCl','H2O2/NaCl']
    
    adaptive_df = df.copy()
    for cond in exp_conds:
        this_source = adaptive_df[adaptive_df['Source_Env']==cond].copy()
        nonadaptive_BCs = this_source[this_source[f'{cond}_fitness']<adapt_thr].index.tolist()
        adaptive_df = adaptive_df.drop(nonadaptive_BCs)
        
    return(adaptive_df)


def plot_adaptive_thr(df,title):
    # Plot adaptive thresholds
    
    exp_conds = ['Glu','Gal','Lac','H2O2','NaCl','Glu/Gal',
             'Glu/Lac','Glu/H2O2','Glu/NaCl','H2O2/NaCl']

    # Plot how adaptive threshold affects results:
    num_adaptive = []
    adapt_thrs = np.linspace(-0.2, 0.5, 100)#[-0.06,-0.04,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.22,0.24,0.26,0.28,0.3]
    for k in adapt_thrs:
        df_thr = df.copy()
        adapt_thr = k
        for cond in exp_conds:
            this_source = df_thr[df_thr['Source_Env']==cond].copy()
            nonadaptive_BCs = this_source[this_source[f'{cond}_fitness']<adapt_thr].index.tolist()
            blank_BCs = this_source[pd.isna(this_source[f'{cond}_fitness'])].index.tolist()
            df_thr = df_thr.drop(nonadaptive_BCs)
            df_thr = df_thr.drop(blank_BCs)
        num_adaptive.append(len(df_thr)/len(df))
            
    # Create the plot
    fig, ax = plt.subplots()
    
    # Plot the data
    ax.plot(adapt_thrs,num_adaptive)
    
    # Set major ticks every 0.1 units and minor ticks every 0.02 units on the x-axis
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.02))
    
    # Add grid lines to the plot at minor tick locations
    ax.grid(which='minor', alpha=0.2)
    
    # Add grid lines to the plot at major tick locations
    ax.grid(which='major', alpha=0.5)
    
    ax.set_xlabel('Adaptive Threshold')
    ax.set_ylabel('Fraction Retained Clones')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'adaptive_threshold_retained_{title}.png',dpi=300)
    
    plt.show()


def sep_gen(df,adapt_thr):
    # Separate generalists
    static_conds = ['Glu','Gal','Lac','H2O2','NaCl'] 
    
    gen_df = df.copy()
    spcs = []
    for cond in static_conds:
        non_gens = gen_df[gen_df[f'{cond}_fitness']<adapt_thr].index.tolist()
        spcs.extend(non_gens)
    spcs = np.unique(spcs)
    gen_df = gen_df.drop(spcs)
    
    return(gen_df)


def add_reps(df,df_static,df_fluct_comp1,df_fluct_comp2,BClist,add_exceptions=False,df_static_exceptions=None,df_fluct_comp2_exceptions=None):
    # Add replicate information
 
    if add_exceptions:
        static_conds = ['Glu','Gal','Lac','NaCl']
        static_conds_exceptions = ['H2O2']

        fluct_conds = ['Glu/Gal','Glu/Lac','Glu/NaCl','H2O2/NaCl','Gal/H2O2','Gal/NaCl','Lac/NaCl']
        fluct_conds_exceptions = ['Glu/H2O2','Gal/Lac','Lac/H2O2']
        
    else:
        static_conds = ['Glu','Gal','Lac','H2O2','NaCl']
        fluct_conds = ['Glu/Gal','Glu/Lac','Glu/H2O2','Glu/NaCl','H2O2/NaCl','Gal/Lac','Gal/H2O2','Gal/NaCl','Lac/H2O2','Lac/NaCl']
        
        
    static_dict = {'Glu':'Glu','Gal':'Gal','Lac':'L','H2O2':'high_H_H','NaCl':'N'}
    fluct_dict = {'Glu/Gal':'Glu-Gal','Glu/Lac':'Glu-L','Glu/H2O2':'high_H_Glu-H',
                  'Glu/NaCl':'Glu-N','H2O2/NaCl':'high_H_H-N','Gal/Lac':'Gal-L',
                  'Gal/H2O2':'high_H_Gal-H','Gal/NaCl':'Gal-N','Lac/H2O2':'high_H_L-H',
                  'Lac/NaCl':'L-N'}
    fluct_comp_dict = {'Glu/Gal':['Glu','Gal'],'Glu/Lac':['Glu','Lac'],'Glu/H2O2':['Glu','H2O2'],
                       'Glu/NaCl':['Glu','NaCl'],'H2O2/NaCl':['H2O2','NaCl'],'Gal/Lac':['Gal','Lac'],
                       'Gal/H2O2':['Gal','H2O2'],'Gal/NaCl':['Gal','NaCl'],
                       'Lac/H2O2':['Lac','H2O2'],'Lac/NaCl':['Lac','NaCl']}
    fluct_comp_dict_names = {'Glu/Gal':['Glu','Gal'],'Glu/Lac':['Glu','L'],'Glu/H2O2':['Glu','high_H_H'],
                       'Glu/NaCl':['Glu','N'],'H2O2/NaCl':['high_H','N'],'Gal/Lac':['Gal','L'],
                       'Gal/H2O2':['Gal','high_H_H'],'Gal/NaCl':['Gal','N'],
                       'Lac/H2O2':['L','high_H_H'],'Lac/NaCl':['L','N']}
    
    
    for cond in static_conds:
        cond_name = static_dict[cond]
        for r in [1,2,3]:
            df[f'{cond}_{r}_fitness'] = np.nan
            df[f'{cond}_{r}_fitness'] = np.nan
            cond_df = df_static[df_static['condition']==cond_name]
            rep_cond_df = cond_df[cond_df['rep']==r]
            for BC in df.index:
                df.loc[BC,f'{cond}_{r}_fitness'] = rep_cond_df.loc[BC,'fitness']
                
    if add_exceptions:
        for cond in static_conds_exceptions:
            cond_name = static_dict[cond]
            for r in [1,2,3]:
                df[f'{cond}_{r}_fitness'] = np.nan
                df[f'{cond}_{r}_fitness'] = np.nan
                cond_df = df_static_exceptions[df_static_exceptions['condition']==cond_name]
                rep_cond_df = cond_df[cond_df['rep']==r]
                for BC in df.index:
                    df.loc[BC,f'{cond}_{r}_fitness'] = rep_cond_df.loc[BC,'fitness']
    
    for cond in fluct_conds:
        cond_name = fluct_dict[cond]
        fluct_comps = fluct_comp_dict[cond]
        fluct_comp_names = fluct_comp_dict_names[cond]
        for c in [0,1]:
            comp = fluct_comps[c]
            comp_name = fluct_comp_names[c]
            for r in [1,2,3]:
                df[f'{comp}_{cond}_{r}_fitness'] = np.nan
                
                if c == 0:
                    rep_cond_df = df_fluct_comp1[df_fluct_comp1['condition']==cond_name]
                    rep_cond_df = rep_cond_df[rep_cond_df['rep']==r]
                elif c == 1:
                    rep_cond_df = df_fluct_comp2[df_fluct_comp2['condition']==cond_name]
                    rep_cond_df = rep_cond_df[rep_cond_df['rep']==r]
                
                for BC in df.index:
                    df.loc[BC,f'{comp}_{cond}_{r}_fitness'] = rep_cond_df.loc[BC,'fitness']
                    
    if add_exceptions:
        for cond in fluct_conds_exceptions:
            cond_name = fluct_dict[cond]
            fluct_comps = fluct_comp_dict[cond]
            fluct_comp_names = fluct_comp_dict_names[cond]
            for c in [0,1]:
                comp = fluct_comps[c]
                comp_name = fluct_comp_names[c]
                for r in [1,2,3]:
                    df[f'{comp}_{cond}_{r}_fitness'] = np.nan
                    
                    if c == 0:
                        rep_cond_df = df_fluct_comp1[df_fluct_comp1['condition']==cond_name]
                        rep_cond_df = rep_cond_df[rep_cond_df['rep']==r]
                    elif c == 1:
                        rep_cond_df = df_fluct_comp2_exceptions[df_fluct_comp2_exceptions['condition']==cond_name]
                        rep_cond_df = rep_cond_df[rep_cond_df['rep']==r]
                    
                    for BC in df.index:
                        df.loc[BC,f'{comp}_{cond}_{r}_fitness'] = rep_cond_df.loc[BC,'fitness']
    return(df)

#%% Import data:
data_path = '../../data/'    

BClist = pd.read_csv(f'{data_path}BClist.csv',index_col=0)

MLE_static = pd.read_csv('data/MLE_naive_calc_static_fits_across_reps_skip_zeros.csv',index_col=0)
MLE_static_H = pd.read_csv('data/MLE_naive_calc_static_fits_across_reps_skip_zeros_H.csv',index_col=0)
MLE_fluct_comp1 = pd.read_csv('data/MLE_naive_fluct_fits_comp1_across_reps.csv',index_col=0)
MLE_fluct_comp2_withT0 = pd.read_csv('data/MLE_naive_fluct_fits_comp2_withT0_across_reps.csv',index_col=0)
MLE_fluct_comp2_withoutT0 = pd.read_csv('data/MLE_naive_fluct_fits_comp2_withoutT0_across_reps.csv',index_col=0)
MLE_fluct_comp2_withT0_exceptions = pd.read_csv('data/MLE_naive_fluct_fits_comp2_withT0_exceptions_across_reps.csv',index_col=0)

MLE_static_reps = pd.read_csv('data/MLE_naive_calc_static_fits_sep_reps_skip_zeros.csv',index_col=0)
MLE_static_reps_H = pd.read_csv('data/MLE_naive_calc_static_fits_sep_reps_skip_zeros_H.csv',index_col=0)
MLE_fluct_comp1_reps = pd.read_csv('data/MLE_naive_fluct_fits_comp1_sep_reps.csv',index_col=0)
MLE_fluct_comp2_reps_withT0 = pd.read_csv('data/MLE_naive_fluct_fits_comp2_withT0_sep_reps.csv',index_col=0)
MLE_fluct_comp2_reps_withoutT0 = pd.read_csv('data/MLE_naive_fluct_fits_comp2_withoutT0_sep_reps.csv',index_col=0)
MLE_fluct_comp2_reps_withT0_exceptions = pd.read_csv('data/MLE_naive_fluct_fits_comp2_withT0_exceptions_sep_reps.csv',index_col=0)

# Regression to mean control (3rd and 6th timepoints only):
R2M_fluct_comp1_reps = pd.read_csv('data/calc_fluct_fits_comp1_sep_reps_R2M_control.csv',index_col=0)
R2M_fluct_comp2_reps = pd.read_csv('data/calc_fluct_fits_comp2_sep_reps_R2M_control.csv',index_col=0)
# AVERAGE REPS and make new dataframes:
BC_col_comp1 = []
condition_col_comp1 = []
fitness_col_comp1 = []
std_col_comp1 = []
std_err_col_comp1 = []
BC_col_comp2 = []
condition_col_comp2 = []
fitness_col_comp2 = []
std_col_comp2 = []
std_err_col_comp2 = []
BC_seqs = np.unique(R2M_fluct_comp1_reps.index.tolist())
conditions = np.unique(R2M_fluct_comp1_reps['condition'].tolist())
for cond in conditions:
    this_cond_comp1 = R2M_fluct_comp1_reps[R2M_fluct_comp1_reps['condition']==cond]
    this_cond_comp2 = R2M_fluct_comp2_reps[R2M_fluct_comp2_reps['condition']==cond]
    for BC in BC_seqs:
        this_BC_cond_comp1 = this_cond_comp1[this_cond_comp1.index==BC]
        average_fit_comp1 = this_BC_cond_comp1['fitness'].mean()
        std_fit_comp1 = this_BC_cond_comp1['fitness'].std()
        sem_fit_comp1 = this_BC_cond_comp1['fitness'].sem()
        
        this_BC_cond_comp2 = this_cond_comp2[this_cond_comp2.index==BC]
        average_fit_comp2 = this_BC_cond_comp2['fitness'].mean()
        std_fit_comp2 = this_BC_cond_comp2['fitness'].std()
        sem_fit_comp2 = this_BC_cond_comp2['fitness'].sem()
        
        BC_col_comp1.append(BC)
        condition_col_comp1.append(cond)
        fitness_col_comp1.append(average_fit_comp1)
        std_col_comp1.append(std_fit_comp1)
        std_err_col_comp1.append(sem_fit_comp1)
        
        BC_col_comp2.append(BC)
        condition_col_comp2.append(cond)
        fitness_col_comp2.append(average_fit_comp2)
        std_col_comp2.append(std_fit_comp2)
        std_err_col_comp2.append(sem_fit_comp2)

R2M_fluct_comp1 = pd.DataFrame({'BC': BC_col_comp1, 'condition': condition_col_comp1, 
                                'fitness': fitness_col_comp1, 'std': std_col_comp1, 'Naive_std_err': std_err_col_comp1})
R2M_fluct_comp1 = R2M_fluct_comp1.set_index('BC')
R2M_fluct_comp2 = pd.DataFrame({'BC': BC_col_comp2, 'condition': condition_col_comp2, 
                                'fitness': fitness_col_comp2, 'std': std_col_comp2, 'Naive_std_err': std_err_col_comp2})
R2M_fluct_comp2 = R2M_fluct_comp2.set_index('BC')


BClist.set_index('Sequence',inplace=True)
BClist_with_BCID = pd.read_csv(f'{data_path}BClist.csv',index_col=0)

source_dict = {'Circ_A1':'Alice','Circ_A2':'Alice','Circ_B1':'Bob',
               'Circ_B2':'Bob','AB-leftovers':'Alice/Bob','Circ_G':'Glu', 
               'SEvo_G':'Glu','Circ_H':'H2O2','Circ_L':'Lac','Circ_N':'NaCl',
               'SEvo_N':'NaCl','SEvo_Gal':'Gal','Circ_HF1':'Glu/H2O2',
               'Circ_HF2':'Glu/H2O2','Circ_HN1':'H2O2/NaCl','Circ_HN2':'H2O2/NaCl',
               'Circ_LF1':'Glu/Lac','Circ_LF2':'Glu/Lac','Circ_NF':'Glu/NaCl',
               'SEvo_GN':'Glu/NaCl','SEvo_GalF':'Glu/Gal','DPY256':'Ancestor'}
BClist['Source_Env'] = ""
for k in BClist.index:
    if BClist.loc[k,'Source'] in source_dict:
        BClist.loc[k,'Source_Env'] = source_dict[BClist.loc[k,'Source']]
    else:
        BClist.loc[k,'Source_Env'] = 'Neutral'

# Add source environments:
MLE_static = add_src_envs(MLE_static,BClist)
MLE_static_H = add_src_envs(MLE_static_H,BClist)
MLE_fluct_comp1 = add_src_envs(MLE_fluct_comp1,BClist) 
MLE_fluct_comp2_withT0 = add_src_envs(MLE_fluct_comp2_withT0,BClist) 
MLE_fluct_comp2_withoutT0 = add_src_envs(MLE_fluct_comp2_withoutT0,BClist)
MLE_fluct_comp2_withT0_exceptions = add_src_envs(MLE_fluct_comp2_withT0_exceptions,BClist)

MLE_static_reps = add_src_envs(MLE_static_reps,BClist)
MLE_static_reps_H = add_src_envs(MLE_static_reps_H,BClist)
MLE_fluct_comp1_reps = add_src_envs(MLE_fluct_comp1_reps,BClist) 
MLE_fluct_comp2_reps_withT0 = add_src_envs(MLE_fluct_comp2_reps_withT0,BClist) 
MLE_fluct_comp2_reps_withoutT0 = add_src_envs(MLE_fluct_comp2_reps_withoutT0,BClist)
MLE_fluct_comp2_reps_withT0_exceptions = add_src_envs(MLE_fluct_comp2_reps_withT0_exceptions,BClist)


R2M_fluct_comp1_reps = add_src_envs(R2M_fluct_comp1_reps,BClist) 
R2M_fluct_comp2_reps = add_src_envs(R2M_fluct_comp2_reps,BClist) 
R2M_fluct_comp1 = add_src_envs(R2M_fluct_comp1,BClist)
R2M_fluct_comp2 = add_src_envs(R2M_fluct_comp2,BClist)

#%% Make non-neutral, non-Alice/Bob datatframes, drop mutants missing some static fitnesses, and save:

MLE_non_neutralAB_withT0_fitness = make_non_neutralAB_df(MLE_static,MLE_fluct_comp1,MLE_fluct_comp2_withT0,BClist)
MLE_non_neutralAB_withT0_fitness = drop_empty_static(MLE_non_neutralAB_withT0_fitness)
MLE_non_neutralAB_withT0_fitness.to_csv('formatted_data/non_neutralAB_fitness_withT0.csv')

MLE_non_neutralAB_withoutT0_fitness = make_non_neutralAB_df(MLE_static,MLE_fluct_comp1,MLE_fluct_comp2_withoutT0,BClist)
MLE_non_neutralAB_withoutT0_fitness = drop_empty_static(MLE_non_neutralAB_withoutT0_fitness)
MLE_non_neutralAB_withoutT0_fitness.to_csv('formatted_data/non_neutralAB_fitness_withoutT0.csv')

# Add exceptions (dropping select timepoints) to dataframes:
MLE_non_neutralAB_withT0_fitness_exceptions = make_non_neutralAB_df(MLE_static,MLE_fluct_comp1,MLE_fluct_comp2_withT0,BClist,
                                                                    True,MLE_static_H,MLE_fluct_comp2_withT0_exceptions)
MLE_non_neutralAB_withT0_fitness_exceptions = drop_empty_static(MLE_non_neutralAB_withT0_fitness_exceptions)
MLE_non_neutralAB_withT0_fitness_exceptions.to_csv('formatted_data/non_neutralAB_fitness_withT0_exceptions.csv')

MLE_non_neutralAB_withoutT0_fitness_exceptions = make_non_neutralAB_df(MLE_static,MLE_fluct_comp1,MLE_fluct_comp2_withoutT0,BClist,
                                                                       True,MLE_static_H,MLE_fluct_comp2_withT0_exceptions)
MLE_non_neutralAB_withoutT0_fitness_exceptions = drop_empty_static(MLE_non_neutralAB_withoutT0_fitness_exceptions)
MLE_non_neutralAB_withoutT0_fitness_exceptions.to_csv('formatted_data/non_neutralAB_fitness_withoutT0_exceptions.csv')

#%%
# R2M (regression to mean) control:
R2M_non_neutralAB_fitness = make_non_neutralAB_df(MLE_static,R2M_fluct_comp1,R2M_fluct_comp2,BClist)
R2M_non_neutralAB_fitness = drop_empty_static(R2M_non_neutralAB_fitness)
R2M_non_neutralAB_fitness.to_csv('formatted_data/non_neutralAB_fitness_R2M.csv')


#%% Plot threshold:

plot_adaptive_thr(MLE_non_neutralAB_withT0_fitness_exceptions,'withT0_exc')
plot_adaptive_thr(MLE_non_neutralAB_withoutT0_fitness_exceptions,'withoutT0_exc')

#%% Separate adaptive:
    
adaptive_non_neutralAB_fitness_withT0 = sep_adaptive(MLE_non_neutralAB_withT0_fitness,0.05)
adaptive_non_neutralAB_fitness_withT0.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withT0.csv')

adaptive_non_neutralAB_fitness_withT0_exceptions = sep_adaptive(MLE_non_neutralAB_withT0_fitness_exceptions,0.05)
adaptive_non_neutralAB_fitness_withT0_exceptions.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withT0_exceptions.csv')

adaptive_non_neutralAB_fitness_withoutT0 = sep_adaptive(MLE_non_neutralAB_withoutT0_fitness,0.05)
adaptive_non_neutralAB_fitness_withoutT0.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withoutT0.csv')

adaptive_non_neutralAB_fitness_withoutT0_exceptions = sep_adaptive(MLE_non_neutralAB_withoutT0_fitness_exceptions,0.05)
adaptive_non_neutralAB_fitness_withoutT0_exceptions.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withoutT0_exceptions.csv')

adaptive_non_neutralAB_fitness_withT0_high = sep_adaptive(MLE_non_neutralAB_withT0_fitness,0.1)
adaptive_non_neutralAB_fitness_withT0_high.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withT0_0.1_threshold.csv')

adaptive_non_neutralAB_fitness_withT0_high_exceptions = sep_adaptive(MLE_non_neutralAB_withT0_fitness_exceptions,0.1)
adaptive_non_neutralAB_fitness_withT0_high_exceptions.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withT0_0.1_threshold_exceptions.csv')

adaptive_non_neutralAB_fitness_withoutT0_high = sep_adaptive(MLE_non_neutralAB_withoutT0_fitness,0.1)
adaptive_non_neutralAB_fitness_withoutT0_high.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withoutT0_0.1_threshold.csv')

adaptive_non_neutralAB_fitness_withoutT0_high_exceptions = sep_adaptive(MLE_non_neutralAB_withoutT0_fitness_exceptions,0.1)
adaptive_non_neutralAB_fitness_withoutT0_high_exceptions.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withoutT0_0.1_threshold_exceptions.csv')

#%%
# R2M control:
adaptive_non_neutralAB_fitness_R2M = sep_adaptive(R2M_non_neutralAB_fitness,0.05)
adaptive_non_neutralAB_fitness_R2M.to_csv('formatted_data/adaptive_non_neutralAB_fitness_R2M.csv')


#%% Add rep info:
    
non_neutralAB_fitness_withT0_with_reps = add_reps(MLE_non_neutralAB_withT0_fitness,
                                                           MLE_static_reps,MLE_fluct_comp1_reps,MLE_fluct_comp2_reps_withT0,BClist)
non_neutralAB_fitness_withT0_with_reps.to_csv('formatted_data/non_neutralAB_fitness_withreps_withT0.csv')

non_neutralAB_fitness_withoutT0_with_reps = add_reps(MLE_non_neutralAB_withoutT0_fitness,
                                                           MLE_static_reps,MLE_fluct_comp1_reps,MLE_fluct_comp2_reps_withoutT0,BClist)
non_neutralAB_fitness_withoutT0_with_reps.to_csv('formatted_data/non_neutralAB_fitness_withreps_withoutT0.csv')

non_neutralAB_fitness_withT0_with_reps_exceptions = add_reps(MLE_non_neutralAB_withT0_fitness_exceptions,
                                                           MLE_static_reps,MLE_fluct_comp1_reps,MLE_fluct_comp2_reps_withT0,BClist,
                                                           True,MLE_static_reps_H,MLE_fluct_comp2_reps_withT0_exceptions)
non_neutralAB_fitness_withT0_with_reps_exceptions.to_csv('formatted_data/non_neutralAB_fitness_withreps_withT0_exceptions.csv')

non_neutralAB_fitness_withoutT0_with_reps_exceptions = add_reps(MLE_non_neutralAB_withoutT0_fitness_exceptions,
                                                           MLE_static_reps,MLE_fluct_comp1_reps,MLE_fluct_comp2_reps_withoutT0,BClist,
                                                           True,MLE_static_reps_H,MLE_fluct_comp2_reps_withT0_exceptions)
non_neutralAB_fitness_withoutT0_with_reps_exceptions.to_csv('formatted_data/non_neutralAB_fitness_withreps_withoutT0_exceptions.csv')

adaptive_non_neutralAB_fitness_withT0_with_reps = add_reps(adaptive_non_neutralAB_fitness_withT0,
                                                           MLE_static_reps,MLE_fluct_comp1_reps,MLE_fluct_comp2_reps_withT0,BClist)
adaptive_non_neutralAB_fitness_withT0_with_reps.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withreps_withT0.csv')
    
adaptive_non_neutralAB_fitness_withoutT0_with_reps = add_reps(adaptive_non_neutralAB_fitness_withoutT0,
                                                           MLE_static_reps,MLE_fluct_comp1_reps,MLE_fluct_comp2_reps_withoutT0,BClist)
adaptive_non_neutralAB_fitness_withoutT0_with_reps.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withreps_withoutT0.csv')


adaptive_non_neutralAB_fitness_withT0_with_reps_exceptions = add_reps(adaptive_non_neutralAB_fitness_withT0_exceptions,
                                                           MLE_static_reps,MLE_fluct_comp1_reps,MLE_fluct_comp2_reps_withT0,BClist,
                                                           True,MLE_static_reps_H,MLE_fluct_comp2_reps_withT0_exceptions)
adaptive_non_neutralAB_fitness_withT0_with_reps_exceptions.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withreps_withT0_exceptions.csv')
    
adaptive_non_neutralAB_fitness_withoutT0_with_reps_exceptions = add_reps(adaptive_non_neutralAB_fitness_withoutT0_exceptions,
                                                           MLE_static_reps,MLE_fluct_comp1_reps,MLE_fluct_comp2_reps_withoutT0,BClist,
                                                           True,MLE_static_reps_H,MLE_fluct_comp2_reps_withT0_exceptions)
adaptive_non_neutralAB_fitness_withoutT0_with_reps_exceptions.to_csv('formatted_data/adaptive_non_neutralAB_fitness_withreps_withoutT0_exceptions.csv')


