#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:10:36 2024

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

#%% Import data:
    
BClist = pd.read_csv('../data/BClist.csv',index_col=0)
fitness_df = pd.read_csv('../data/fitness_df.csv',index_col=0)
counts_df = pd.read_csv('../data/tidy_bc_fitness_counts.csv',index_col=0)

# Make dataframe of counts converted to frequencies:
freq_df = counts_df.copy()
freq_df['frequency'] = 0.0
conds = np.unique(counts_df['condition']).tolist()
all_BCs = np.unique(counts_df.index).tolist()
for cond in conds:
    freq_cond_df = freq_df[freq_df['condition']==cond]
    for t in np.unique(counts_df['time']).tolist():
        if t in np.unique(freq_cond_df['time']).tolist():
            total_counts = np.sum(freq_cond_df[freq_cond_df['time']==t]['count'])
            for BC in all_BCs:
                freq_df[(freq_df['condition']==cond) & (freq_df['time']==t)].loc[BC,'frequency'] = freq_df[(freq_df['condition']==cond) & 
                                                                                                (freq_df['time']==t)].loc[BC,'count']/total_counts
    
    

#%% Fig. 2A:

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
plt.figure(figsize=(25,10))

fluct_conds = [Gal_conds,L_conds,HN_conds]
low_lims = [-0.7,-0.7,-1.1]
high_lims = [0.7,1,1.7]
# Choose mutants to highlight:
all_BCs = [[516,939,1369],
       [586,838,1114],
       [548,759,1110]]
mut_colors = ['magenta','blue','lime']
conds_title_dict = {'Glu/Gal':'Glu / Gal','Glu/Lac':'Glu / Lac','H2O2/NaCl':'H$_2$O$_2$ / NaCl'}

pl=0
plt.figure(figsize=(15,5))
for conds in fluct_conds:#pair_conds:
    
    BCs = all_BCs[pl]
    df = fitness_df.copy()
    df['Predicted Fluctuating Fitness'] = ""
    df['Predicted Fluctuating Fitness'] = df[f'{conds[0]}_fitness']*(.5)+df[f'{conds[1]}_fitness']*(.5)
    df['Predicted Fluctuating Error'] = 0.5*np.sqrt(df[f'{conds[0]}_SEM_bootstrap']**2+df[f'{conds[1]}_SEM_bootstrap']**2)

    plt.subplot(1,3,pl+1)
    plt.scatter(df['Predicted Fluctuating Fitness'],df[f'{conds[2]}_fitness'],color='darkgray',alpha=0.4)
    for k in range(3):
        BC = BCs[k]
        BC_seq = BClist.loc[BC,'Sequence']
        plt.scatter(df.loc[BC_seq,'Predicted Fluctuating Fitness'],df.loc[BC_seq,f'{conds[2]}_fitness'],color=mut_colors[k])
        plt.errorbar(df.loc[BC_seq,'Predicted Fluctuating Fitness'],df.loc[BC_seq,f'{conds[2]}_fitness'],
                     xerr=df.loc[BC_seq,'Predicted Fluctuating Error'], yerr=df.loc[BC_seq,f'{conds[2]}_SEM_bootstrap'],color=mut_colors[k])

    if pl==0:
        plt.ylabel('Fluctuating fitness (per cycle)')
    plt.xlabel('Average static fitness')  
    
    plt.title(conds_title_dict[conds[2]])
    plt.xlim(low_lims[pl],high_lims[pl])
    plt.ylim(low_lims[pl],high_lims[pl])
    xpoints = plt.xlim()
    plt.plot(xpoints,xpoints,'k--') # identity line
    pl+=1
plt.tight_layout()


#%% Fig. 2B-D:
    
#%% functions:
    
def average_trajectories(df,cond,BC,times,reps):
    # Generate average trajectory across replicates
    aves=[]
    start = np.mean([df[f'{cond}-{r}-T{times[0]}'].loc[BC] for r in reps])
    aves.append(start)
    mean_slope = []
    for k in range(len(times)-1):
        this_time = times[k]
        next_time = times[k+1]
        slopes = [df[f'{cond}-{r}-T{next_time}'].loc[BC] - df[f'{cond}-{r}-T{this_time}'].loc[BC] for r in reps]
        mean_slope.append(np.mean(slopes))
        aves.append(start+np.mean(slopes))
        start = start+np.mean(slopes)
    return(aves)

def get_mean_fit(df,cond,timept,reps):
    # Get mean fitness for a particular condition and timepoint, across all neutrals and replicates
    # NOTE: df must include only neutral frequencies
    neut_change = 0
    start_timept = timept-1
    for r in reps:
        neut_change += np.mean(-np.log(df[f'{cond}-{r}-T{timept}']/df[f'{cond}-{r}-T{start_timept}']))
    mean_neut_change = neut_change/(len(reps))
    return(mean_neut_change)

def exp_trajectory(times,start_freq,fits,mean_fits):
    # Generate expected trajectories based on starting frequency, fitnesses, and mean fitnesses across timepoints
    exp_traj = []
    exp_traj.append(start_freq)
    this_freq = start_freq
    for k in range(len(times)-1):
        next_freq = this_freq*np.exp(fits[k]-mean_fits[k])
        exp_traj.append(next_freq)
        this_freq = next_freq
    return(exp_traj)
    
def plot_three_conds_nine_BCs(df_combined_freq,df_neutral_freq,fluct_conds,all_BCs,all_BC_seqs,plot_obs_trajectory=True,uniform_scale=False,log_scale=False,choose_colors=['k','k','k'],grayscale=False):

    pl=0
    plt.figure(figsize=(15,10))
    
    for b in range(3):
        BCs = all_BCs[b]
        BC_seqs = all_BC_seqs[b]
        c = 0
        for fluct_cond in fluct_conds:
            BC = BCs[c]
            print(BC)
            BC_seq = BC_seqs[c]
            fluct_pair = fluct_cond.split('/')*3
            
            cond = fluct_cond
            if cond == 'Gal/Lac':
                reps = [1,2]
            else:
                reps = [1,2,3]
            k=1
            
            fluct_timepts = [1,2,3,4,5]
            
            plt.subplot(3,3,pl+1)
            ave_freqs = average_trajectories(df_combined_freq,cond,BC,fluct_timepts,reps)
            if plot_obs_trajectory:
                plt.plot([n+1 for n in fluct_timepts],ave_freqs,color='k')
            
            fits = []
            mean_fits=[]
            for k in range(len(fluct_timepts)-1):
                mean_fits.append(get_mean_fit(df_neutral_freq,cond,fluct_timepts[k+1],reps))
                env = fluct_pair[fluct_timepts[k+1]]
                fits.append(fitness_df[f'{env}_fitness'].loc[BC_seq])
            start_freq = ave_freqs[0]
            exp_traj = exp_trajectory(fluct_timepts,start_freq,fits,mean_fits)
            
            plt.plot([n+1 for n in fluct_timepts],exp_traj,'k--')
            
            if grayscale:
                plt.axvspan(2, 3, alpha=0.3, color=color_dict_gray[fluct_pair[0]])
                plt.axvspan(4, 5, alpha=0.3, color=color_dict_gray[fluct_pair[0]])
                plt.axvspan(3, 4, alpha=0.3, color=color_dict_gray[fluct_pair[1]])
                plt.axvspan(5, 6, alpha=0.3, color=color_dict_gray[fluct_pair[1]])
                
            else:
                plt.axvspan(2, 3, alpha=0.3, color=color_dict[fluct_pair[0]])
                plt.axvspan(4, 5, alpha=0.3, color=color_dict[fluct_pair[0]])
                plt.axvspan(3, 4, alpha=0.3, color=color_dict[fluct_pair[1]])
                plt.axvspan(5, 6, alpha=0.3, color=color_dict[fluct_pair[1]])
        
            if log_scale:
                plt.yscale('log')
            if uniform_scale:
                plt.ylim([1.15e-3,2.1e-3])
                
            ax = plt.gca()
            # Set the spines properties
            ax.spines['left'].set_color(choose_colors[b])
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_color(choose_colors[b])
            ax.spines['right'].set_linewidth(3)
            ax.spines['bottom'].set_color(choose_colors[b])
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['top'].set_color(choose_colors[b])
            ax.spines['top'].set_linewidth(3)
            
            if b==2:
                plt.xlabel('Timepoint')
            plt.xticks([x+1 for x in fluct_timepts])
            if c == 0:
                plt.ylabel('Frequency in pool')
                
            if pl == 4:  # This is the index of the subplot in question (second row, second column)
                ax.yaxis.set_major_locator(FixedLocator([1e-2, 3e-2]))
                ax.set_yticklabels([r'$10^{-2}$', r'$3\times 10^{-2}$'])
                #plt.locator_params(axis='y', nbins=2)
                #plt.ylim([1e-3,6e-2])
                #tick_values = np.logspace(np.log10(1e-2), np.log10(5e-2), num=2)
                #ax.set_yticks(tick_values)
                #ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation))
                
            
            pl+=1
            c+=1
        
    #plt.title(f'{cond}')
    plt.tight_layout()
    plt.show()
    

#%% Plot all, gray scale:
fluct_conds = ['Glu/Gal','Glu/Lac','H2O2/NaCl']
all_BCs = [[1369,1114,1110],[516,586,548],[939,838,759]]  # different order than above but same mutants
BC_seqs_1 = [BClist.loc[BC,'Sequence'] for BC in all_BCs[0]]
BC_seqs_2 = [BClist.loc[BC,'Sequence'] for BC in all_BCs[1]]
BC_seqs_3 = [BClist.loc[BC,'Sequence'] for BC in all_BCs[2]]
all_BC_seqs = [BC_seqs_1,BC_seqs_2,BC_seqs_3]
choose_colors = ['lime','magenta','blue']
plot_obs_trajectory=True
uniform_scale=False
log_scale=True
grayscale=True
    
plot_three_conds_nine_BCs(df_combined_freq,df_neutral_freq,fluct_conds,all_BCs,all_BC_seqs,plot_obs_trajectory,uniform_scale,log_scale,choose_colors,grayscale)
