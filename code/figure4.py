#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:07:54 2024

@author: clare
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import random

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

#%% Import data:
BClist = pd.read_csv('../data/BClist.csv',index_col=0)
fitness_df = pd.read_csv('../data/fitness_df.csv',index_col=0)
fitness_df_withreps = pd.read_csv('../code/infer_fitness/formatted_data/adaptive_non_neutralAB_fitness_withreps_withoutT0_exceptions.csv',index_col=0)

# High-variance mutants:
df_high_rep_variation_mutants = pd.read_csv('../data/high_rep_variation_mutants.csv',index_col=0)
high_rep_variation_mutants = df_high_rep_variation_mutants.index.tolist()
fitness_df_dropped = fitness_df.drop(high_rep_variation_mutants)
fitness_df_withreps_dropped = fitness_df_withreps.drop(high_rep_variation_mutants)
# COMMENT OUT IF NOT DROPPING HIGH VARIANCE MUTANTS:
fitness_df = fitness_df_dropped
fitness_df_withreps = fitness_df_withreps_dropped

#%% Figure 4A (data summary):
    
plt.figure(figsize=(5.2, 5))
# For Mean_Nonadditivity vs Mean_Diff_Static
x1 = fitness_df['Mean_Diff_Static'].values
y1 = fitness_df['Mean_Nonadditivity'].values
slope1 = np.linalg.lstsq(x1[:, np.newaxis], y1, rcond=None)[0][0]
line_x1 = np.linspace(np.min(x1), np.max(x1), 100)
line_y1 = slope1 * line_x1
plt.scatter(x1, y1, alpha=0.2, c='salmon')
plt.plot(line_x1, line_y1, 'salmon', label='Nonadditivity')

# For Mean_Memory vs Mean_Diff_Static
x2 = fitness_df['Mean_Diff_Static'].values
y2 = fitness_df['Mean_Memory'].values
slope2 = np.linalg.lstsq(x2[:, np.newaxis], y2, rcond=None)[0][0]
line_x2 = np.linspace(np.min(x2), np.max(x2), 100)
line_y2 = slope2 * line_x2
plt.scatter(x2, y2, alpha=0.2, c='purple')#'steelblue')
plt.plot(line_x2, line_y2, 'purple', label='Memory')
plt.ylim([0,1.2])
plt.xlim([0,1.4])

plt.xlabel('Mean fitness difference (static)')
plt.ylabel('Mean fitness change (fluctuating)')
plt.tight_layout()
    
#%% Figure 4B:
    
plt.figure(figsize=(15,10))

conds = [['Glu','Gal'],['Glu','Lac'],['Gal','Lac'],['Glu','H2O2'],['Glu','NaCl'],['H2O2','NaCl'],['Gal','H2O2'],
         ['Gal','NaCl'],['Lac','NaCl']]#,['Lac','H2O2']]

home_better=0
away_better=0
home_much_better=0
borderline=0
away_much_better=0
border_thr = 0.05  # Define a threshold beyond which home or away environment is better at prediction

Reversal_BCs = {}
Reversal_srcs = {}

for cond in conds:
    first = cond[0]
    second = cond[1]
    fluct = f'{first}/{second}'

    # First component:
    home_err = np.abs(fitness_df[f'{first}_{fluct}_fitness']-fitness_df[f'{first}_fitness'])
    away_err = np.abs(fitness_df[f'{first}_{fluct}_fitness']-fitness_df[f'{second}_fitness'])
    diff = home_err-away_err
    home_better += len(diff[diff<0])
    away_better += len(diff[diff>0])
    
    home_much_better += len(diff[diff<-border_thr])
    away_much_better += len(diff[diff>border_thr])
    borderline += len([x for x in diff if -border_thr < x < border_thr])
    
    reversal_BCs = diff[diff>border_thr].index.tolist()
    srcs = []
    for bc in reversal_BCs:
        srcs.append(fitness_df[fitness_df.index==bc]['Source_Env'].tolist()[0])
    Reversal_BCs[f'{first}_{fluct}'] = reversal_BCs
    Reversal_srcs[f'{first}_{fluct}'] = srcs
    
    # Second component:
    home_err = np.abs(fitness_df[f'{second}_{fluct}_fitness']-fitness_df[f'{second}_fitness'])
    away_err = np.abs(fitness_df[f'{second}_{fluct}_fitness']-fitness_df[f'{first}_fitness'])
    diff = home_err-away_err
    home_better += len(diff[diff<0])
    away_better += len(diff[diff>0])
    
    home_much_better += len(diff[diff<-border_thr])
    away_much_better += len(diff[diff>border_thr])
    borderline += len([x for x in diff if -border_thr < x < border_thr])
    
    reversal_BCs = diff[diff>border_thr].index.tolist()
    srcs = []
    for bc in reversal_BCs:
        srcs.append(fitness_df[fitness_df.index==bc]['Source_Env'].tolist()[0])
    Reversal_BCs[f'{second}_{fluct}'] = reversal_BCs
    Reversal_srcs[f'{second}_{fluct}'] = srcs

    
frac_home_better = home_better/(home_better+away_better)
frac_away_better = away_better/(home_better+away_better)

frac_home_much_better = home_much_better/(home_much_better+away_much_better+borderline)
frac_away_much_better = away_much_better/(home_much_better+away_much_better+borderline)
frac_borderline = borderline/(home_much_better+away_much_better+borderline)

home_better=0
away_better=0
home_much_better=0
borderline=0
away_much_better=0
for cond in conds:
        
    median = fitness_df['Mean_Diff_Static'].quantile(0.9)
    df_high = fitness_df[fitness_df['Mean_Diff_Static'] > median]
    
    first = cond[0]
    second = cond[1]
    fluct = f'{first}/{second}'

    home_err = np.abs(df_high[f'{first}_{fluct}_fitness']-df_high[f'{first}_fitness'])
    away_err = np.abs(df_high[f'{first}_{fluct}_fitness']-df_high[f'{second}_fitness'])
    diff = home_err-away_err
    home_better += len(diff[diff<0])
    away_better += len(diff[diff>0])
    
    home_much_better += len(diff[diff<-border_thr])
    away_much_better += len(diff[diff>border_thr])
    borderline += len([x for x in diff if -border_thr < x < border_thr])

    
    home_err = np.abs(df_high[f'{second}_{fluct}_fitness']-df_high[f'{second}_fitness'])
    away_err = np.abs(df_high[f'{second}_{fluct}_fitness']-df_high[f'{first}_fitness'])
    diff = home_err-away_err
    home_better += len(diff[diff<0])
    away_better += len(diff[diff>0])
    
    home_much_better += len(diff[diff<-border_thr])
    away_much_better += len(diff[diff>border_thr])
    borderline += len([x for x in diff if -border_thr < x < border_thr])

    
frac_home_better_highdiff = home_better/(home_better+away_better)
frac_away_better_highdiff = away_better/(home_better+away_better)

frac_home_much_better_highdiff = home_much_better/(home_much_better+away_much_better+borderline)
frac_away_much_better_highdiff = away_much_better/(home_much_better+away_much_better+borderline)
frac_borderline_highdiff = borderline/(home_much_better+away_much_better+borderline)

home_better=0
away_better=0
home_much_better=0
borderline=0
away_much_better=0
for cond in conds:
        
    median = fitness_df['Mean_Diff_Static'].quantile(0.1)
    df_low = fitness_df[fitness_df['Mean_Diff_Static'] < median]
    
    first = cond[0]
    second = cond[1]
    fluct = f'{first}/{second}'

    home_err = np.abs(df_low[f'{first}_{fluct}_fitness']-df_low[f'{first}_fitness'])
    away_err = np.abs(df_low[f'{first}_{fluct}_fitness']-df_low[f'{second}_fitness'])
    diff = home_err-away_err
    home_better += len(diff[diff<0])
    away_better += len(diff[diff>0])
    
    home_much_better += len(diff[diff<-border_thr])
    away_much_better += len(diff[diff>border_thr])
    borderline += len([x for x in diff if -border_thr < x < border_thr])

    
    home_err = np.abs(df_low[f'{second}_{fluct}_fitness']-df_low[f'{second}_fitness'])
    away_err = np.abs(df_low[f'{second}_{fluct}_fitness']-df_low[f'{first}_fitness'])
    diff = home_err-away_err
    home_better += len(diff[diff<0])
    away_better += len(diff[diff>0])
    
    home_much_better += len(diff[diff<-border_thr])
    away_much_better += len(diff[diff>border_thr])
    borderline += len([x for x in diff if -border_thr < x < border_thr])

frac_home_better_lowdiff = home_better/(home_better+away_better)
frac_away_better_lowdiff = away_better/(home_better+away_better)

frac_home_much_better_lowdiff = home_much_better/(home_much_better+away_much_better+borderline)
frac_away_much_better_lowdiff = away_much_better/(home_much_better+away_much_better+borderline)
frac_borderline_lowdiff = borderline/(home_much_better+away_much_better+borderline)

labels = ['All data','High fitness\ndifference','Low fitness\ndifference']
group1 = [frac_home_much_better,frac_home_much_better_highdiff,frac_home_much_better_lowdiff]
group2 = [frac_borderline,frac_borderline_highdiff,frac_borderline_lowdiff]
group3 = [frac_away_much_better,frac_away_much_better_highdiff,frac_away_much_better_lowdiff]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

plt.figure(figsize=(6,5))
fig, ax = plt.subplots(figsize=(6,5))

bar3 = ax.bar(x, group3, label='Other static environment', color='goldenrod')
bar2 = ax.bar(x, group2, bottom=group3, label='Borderline', color='lightgray')
bar1 = ax.bar(x, group1, bottom=np.array(group2) + np.array(group3), label='Same static environment', color = 'lightblue')

ax.set_ylabel('Fraction')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim([0,1])
fig.tight_layout()


#%% Figure 4C:

# Controls: predicting fluct reps with fluct reps:
    
plt.figure(figsize=(15,10))

conds = [['Glu','Gal'],['Glu','Lac'],['Gal','Lac'],['Glu','H2O2'],['Glu','NaCl'],['H2O2','NaCl'],['Gal','H2O2'],
         ['Gal','NaCl'],['Lac','NaCl']]#,['Lac','H2O2']]

home_better=0
away_better=0
home_much_better=0
borderline=0
away_much_better=0
replicates = [1,2,3]

for cond in conds:
    first = cond[0]
    second = cond[1]
    fluct = f'{first}/{second}'

    for r, (r1, r2) in enumerate(combinations(replicates, 2)):
        this_r = [i for i in replicates if i not in (r1, r2)][0]
        
        home_err_1 = np.abs(fitness_df_withreps[f'{first}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{fluct}_{r1}_fitness'])
        home_err_2 = np.abs(fitness_df_withreps[f'{first}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{fluct}_{r2}_fitness'])
        
        away_err_1 = np.abs(fitness_df_withreps[f'{first}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{fluct}_{r1}_fitness'])
        away_err_2 = np.abs(fitness_df_withreps[f'{first}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{fluct}_{r2}_fitness'])
        away_err_3 = np.abs(fitness_df_withreps[f'{first}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{fluct}_{this_r}_fitness'])
        
        diff_1 = home_err_1-away_err_1
        diff_2 = home_err_2-away_err_2
        diff_12 = home_err_1-away_err_2
        diff_21 = home_err_2-away_err_1
        diff_13 = home_err_1-away_err_3
        diff_23 = home_err_2-away_err_3
        
        home_better += len(diff_1[diff_1<0]) + len(diff_2[diff_2<0]) + len(diff_12[diff_12<0]) + len(diff_21[diff_21<0]) \
                        + len(diff_13[diff_13<0]) + len(diff_23[diff_23<0])
        away_better += len(diff_1[diff_1>0]) + len(diff_2[diff_2>0]) + len(diff_12[diff_12>0]) + len(diff_21[diff_21>0]) \
                        + len(diff_13[diff_13>0]) + len(diff_23[diff_23>0])
        
        for diff in [diff_1,diff_2,diff_12,diff_21,diff_13,diff_23]:
            home_much_better += len(diff[diff<-border_thr])
            away_much_better += len(diff[diff>border_thr])
            borderline += len([x for x in diff if -border_thr < x < border_thr])
        
        home_err_1 = np.abs(fitness_df_withreps[f'{second}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{fluct}_{r1}_fitness'])
        home_err_2 = np.abs(fitness_df_withreps[f'{second}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{fluct}_{r2}_fitness'])
        
        away_err_1 = np.abs(fitness_df_withreps[f'{second}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{fluct}_{r1}_fitness'])
        away_err_2 = np.abs(fitness_df_withreps[f'{second}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{fluct}_{r2}_fitness'])
        away_err_3 = np.abs(fitness_df_withreps[f'{second}_{fluct}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{fluct}_{this_r}_fitness'])
        
        diff_1 = home_err_1-away_err_1
        diff_2 = home_err_2-away_err_2
        diff_12 = home_err_1-away_err_2
        diff_21 = home_err_2-away_err_1
        diff_13 = home_err_1-away_err_3
        diff_23 = home_err_2-away_err_3
        
        home_better += len(diff_1[diff_1<0]) + len(diff_2[diff_2<0]) + len(diff_12[diff_12<0]) + len(diff_21[diff_21<0]) \
                        + len(diff_13[diff_13<0]) + len(diff_23[diff_23<0])
        away_better += len(diff_1[diff_1>0]) + len(diff_2[diff_2>0]) + len(diff_12[diff_12>0]) + len(diff_21[diff_21>0]) \
                        + len(diff_13[diff_13>0]) + len(diff_23[diff_23>0])
        
        for diff in [diff_1,diff_2,diff_12,diff_21,diff_13,diff_23]:
            home_much_better += len(diff[diff<-border_thr])
            away_much_better += len(diff[diff>border_thr])
            borderline += len([x for x in diff if -border_thr < x < border_thr])

    
frac_home_better_fluctreps = home_better/(home_better+away_better)
frac_away_better_fluctreps = away_better/(home_better+away_better)

frac_home_much_better_fluctreps = home_much_better/(home_much_better+away_much_better+borderline)
frac_away_much_better_fluctreps = away_much_better/(home_much_better+away_much_better+borderline)
frac_borderline_fluctreps = borderline/(home_much_better+away_much_better+borderline)


# Controls: predicting static replicates with static replicates
    
plt.figure(figsize=(15,10))

conds = [['Glu','Gal'],['Glu','Lac'],['Gal','Lac'],['Glu','H2O2'],['Glu','NaCl'],['H2O2','NaCl'],['Gal','H2O2'],
         ['Gal','NaCl'],['Lac','NaCl']]#,['Lac','H2O2']]

home_better=0
away_better=0
home_much_better=0
borderline=0
away_much_better=0

replicates = [1,2,3]

for cond in conds:
    first = cond[0]
    second = cond[1]
    fluct = f'{first}/{second}'

    for r, (r1, r2) in enumerate(combinations(replicates, 2)):
        this_r = [i for i in replicates if i not in (r1, r2)][0]
        
        home_err_1 = np.abs(fitness_df_withreps[f'{first}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{r1}_fitness'])
        home_err_2 = np.abs(fitness_df_withreps[f'{first}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{r2}_fitness'])
        
        away_err_1 = np.abs(fitness_df_withreps[f'{first}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{r1}_fitness'])
        away_err_2 = np.abs(fitness_df_withreps[f'{first}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{r2}_fitness'])
        away_err_3 = np.abs(fitness_df_withreps[f'{first}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{this_r}_fitness'])
        
        diff_1 = home_err_1-away_err_1
        diff_2 = home_err_2-away_err_2
        diff_12 = home_err_1-away_err_2
        diff_21 = home_err_2-away_err_1
        diff_13 = home_err_1-away_err_3
        diff_23 = home_err_2-away_err_3
        
        home_better += len(diff_1[diff_1<0]) + len(diff_2[diff_2<0]) + len(diff_12[diff_12<0]) + len(diff_21[diff_21<0]) \
                        + len(diff_13[diff_13<0]) + len(diff_23[diff_23<0])
        away_better += len(diff_1[diff_1>0]) + len(diff_2[diff_2>0]) + len(diff_12[diff_12>0]) + len(diff_21[diff_21>0]) \
                        + len(diff_13[diff_13>0]) + len(diff_23[diff_23>0])
                        
        for diff in [diff_1,diff_2,diff_12,diff_21,diff_13,diff_23]:
            home_much_better += len(diff[diff<-border_thr])
            away_much_better += len(diff[diff>border_thr])
            borderline += len([x for x in diff if -border_thr < x < border_thr])
        
        home_err_1 = np.abs(fitness_df_withreps[f'{second}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{r1}_fitness'])
        home_err_2 = np.abs(fitness_df_withreps[f'{second}_{this_r}_fitness']-fitness_df_withreps[f'{second}_{r2}_fitness'])
        
        away_err_1 = np.abs(fitness_df_withreps[f'{second}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{r1}_fitness'])
        away_err_2 = np.abs(fitness_df_withreps[f'{second}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{r2}_fitness'])
        away_err_3 = np.abs(fitness_df_withreps[f'{second}_{this_r}_fitness']-fitness_df_withreps[f'{first}_{this_r}_fitness'])
        
        diff_1 = home_err_1-away_err_1
        diff_2 = home_err_2-away_err_2
        diff_12 = home_err_1-away_err_2
        diff_21 = home_err_2-away_err_1
        diff_13 = home_err_1-away_err_3
        diff_23 = home_err_2-away_err_3
        
        home_better += len(diff_1[diff_1<0]) + len(diff_2[diff_2<0]) + len(diff_12[diff_12<0]) + len(diff_21[diff_21<0]) \
                        + len(diff_13[diff_13<0]) + len(diff_23[diff_23<0])
        away_better += len(diff_1[diff_1>0]) + len(diff_2[diff_2>0]) + len(diff_12[diff_12>0]) + len(diff_21[diff_21>0]) \
                        + len(diff_13[diff_13>0]) + len(diff_23[diff_23>0])
                        
        for diff in [diff_1,diff_2,diff_12,diff_21,diff_13,diff_23]:
            home_much_better += len(diff[diff<-border_thr])
            away_much_better += len(diff[diff>border_thr])
            borderline += len([x for x in diff if -border_thr < x < border_thr])

    
frac_home_better_staticreps = home_better/(home_better+away_better)
frac_away_better_staticreps = away_better/(home_better+away_better)

frac_home_much_better_staticreps = home_much_better/(home_much_better+away_much_better+borderline)
frac_away_much_better_staticreps = away_much_better/(home_much_better+away_much_better+borderline)
frac_borderline_staticreps = borderline/(home_much_better+away_much_better+borderline)
    
# Plot both controls:
    
labels = ['Within static\nreplicates','Within fluctuating\nreplicates']
group1 = [frac_home_much_better_staticreps,frac_home_much_better_fluctreps]
group2 = [frac_borderline_staticreps,frac_borderline_fluctreps]
group3 = [frac_away_much_better_staticreps,frac_away_much_better_fluctreps]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

plt.figure(figsize=(4.95,5))
fig, ax = plt.subplots(figsize=(4.95,5))
bar3 = ax.bar(x, group3, label='Other static environment', color='goldenrod')
bar2 = ax.bar(x, group2, bottom=group3, label='Borderline', color='lightgray')
bar1 = ax.bar(x, group1, bottom=np.array(group2) + np.array(group3), label='Same static environment', color = 'lightblue')

ax.set_ylabel('Fraction')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim([0,1])

fig.tight_layout()
plt.show()


#%% Figure 4D-E:
    
# MODEL: Assign lag advantages in static envs:
    
num_muts = 700
lag_advs = np.zeros([num_muts,5])

# Set ancestor growth rates across five static environments:
r_anc = np.random.uniform(0.3, 0.5, 5)

# Pick home environments 
home_envs = [random.randint(0, 4) for _ in range(num_muts)]

# Pick lag advantages in home envs:
for k in range(num_muts):
    home_lag_adv = np.random.uniform(0.5,3)
    mean = 0.5*home_lag_adv
    SD = 0.5*home_lag_adv
    lag_advs[k] = np.random.normal(mean,SD,5)
    lag_advs[k,home_envs[k]] = home_lag_adv

fitnesses = r_anc*lag_advs

fit_diffs = np.zeros(num_muts)
for k in range(num_muts):
    this_mut_fits = fitnesses[k].tolist()
    # generate all unique pairs
    fit_pairs = list(combinations(this_mut_fits, 2))
    # calculate the difference for each pair and take the absolute value
    differences = [abs(i-j) for i,j in fit_pairs]
    # compute the mean of the differences
    mean_difference = np.mean(differences)
    fit_diffs[k] = mean_difference
    
#%% Move lag disadvantages in fluct envs based on distance to mean:
    
# Move lag adv by some fraction of distance to mean:
lag_factor = 0.8

# Fluct pairs:    
pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]    

all_fluct_lags = []
all_fluct_errs = np.zeros([num_muts,20])
all_opp_fluct_errs = np.zeros([num_muts,20])
home_better=0
away_better=0
track_home_better = np.zeros([num_muts,20])
track_away_better = np.zeros([num_muts,20])
track_nonadd = np.zeros([num_muts,10])
track_mem = np.zeros([num_muts,10])
for k in range(num_muts):
    fluct_lags=[]
    fluct_errs=[]
    opp_fluct_errs=[]
    p=0
    for pair in pairs:
        mean_adv = np.mean(lag_advs[k])
        env1 = pair[0]; env2=pair[1]
        lag1 = lag_advs[k,env1]; lag2 = lag_advs[k,env2]
        fluct_lag1 = lag1 + lag_factor*(mean_adv-lag1)
        fluct_lag2 = lag2 + lag_factor*(mean_adv-lag2)
        fluct_lags.append([fluct_lag1,fluct_lag2])
        fluct_fit1 = r_anc[env1]*fluct_lag1
        fluct_fit2 = r_anc[env2]*fluct_lag2
        nonadd = np.abs((fluct_fit1 + fluct_fit2)/2 - (fitnesses[k,env1] + fitnesses[k,env2])/2)
        track_nonadd[k,p] = nonadd
        fluct_err1 = np.abs(fitnesses[k,env1]-fluct_fit1)
        fluct_err2 = np.abs(fitnesses[k,env2]-fluct_fit2)
        track_mem[k,p] = 0.5*(fluct_err1+fluct_err2)
        opp_fluct_err1 = np.abs(fitnesses[k,env2]-fluct_fit1)
        opp_fluct_err2 = np.abs(fitnesses[k,env1]-fluct_fit2)
        if fluct_err1 < opp_fluct_err1:
            home_better += 1
            track_home_better[k,2*p] = 1
        else:
            away_better += 1
            track_away_better[k,2*p] = 1
            
        if fluct_err2 < opp_fluct_err2:
            home_better += 1
            track_home_better[k,2*p+1] = 1
        else:
            away_better += 1
            track_away_better[k,2*p+1] = 1
        p+=1
        fluct_errs.extend([fluct_err1,fluct_err2])
        opp_fluct_errs.extend([opp_fluct_err1,opp_fluct_err2])
    all_fluct_errs[k] = fluct_errs
    all_opp_fluct_errs[k] = opp_fluct_errs

# Separate into low and high fitness difference:

home_better_list = np.sum(track_home_better,1).tolist()
away_better_list = np.sum(track_away_better,1).tolist()
data = {'Mean_Diff': fit_diffs.tolist(), 'Home_better_total': home_better_list, 'Away_better_total': away_better_list, 
        'Nonadditivity':np.mean(track_nonadd,1).tolist(), 'Memory':np.mean(track_mem,1).tolist()}
df_diff_err = pd.DataFrame(data)

plt.figure(figsize=(5.2,5))
plt.scatter(df_diff_err.Mean_Diff,df_diff_err.Nonadditivity,alpha=0.2,c='salmon')
x1 = df_diff_err['Mean_Diff'].values
y1 = df_diff_err['Nonadditivity'].values
#overall_slope, intercept, r_value, overall_p_value, std_err = stats.linregress(x,y)
#line_x = np.linspace(np.min(x), np.max(x),100)#np.arange(np.min(x), np.max(x))
#line_y = overall_slope*line_x + intercept
slope1 = np.linalg.lstsq(x1[:, np.newaxis], y1, rcond=None)[0][0]
line_x1 = np.linspace(np.min(x1), np.max(x1), 100)
line_y1 = slope1 * line_x1
plt.plot(line_x1, line_y1,'salmon')

plt.scatter(df_diff_err.Mean_Diff,df_diff_err.Memory,alpha=0.2,c='purple')
x2 = df_diff_err['Mean_Diff'].values
y2 = df_diff_err['Memory'].values
slope2 = np.linalg.lstsq(x2[:, np.newaxis], y2, rcond=None)[0][0]
line_x2 = np.linspace(np.min(x2), np.max(x2), 100)
line_y2 = slope2 * line_x2
plt.scatter(x2, y2, alpha=0.2, c='purple')
plt.plot(line_x2, line_y2, 'purple', label='Memory')
plt.ylim([0,1])

plt.xlabel('Mean fitness difference (static)')
plt.ylabel('Mean fitness change (fluctuating)')
plt.tight_layout()



#%% Model bar plot:
    
#Re-run simulation with lots of mutants:
num_muts = 10000
lag_advs = np.zeros([num_muts,5])

# Set ancestor growth rates across five static environments:
r_anc = np.random.uniform(0.3, 0.5, 5)

# Pick home environments 
home_envs = [random.randint(0, 4) for _ in range(num_muts)]

# Pick lag advantages in home envs:
for k in range(num_muts):
    home_lag_adv = np.random.uniform(0.5,3)
    mean = 0.5*home_lag_adv
    SD = 0.5*home_lag_adv
    lag_advs[k] = np.random.normal(mean,SD,5)
    lag_advs[k,home_envs[k]] = home_lag_adv

fitnesses = r_anc*lag_advs

fit_diffs = np.zeros(num_muts)
for k in range(num_muts):
    this_mut_fits = fitnesses[k].tolist()
    # generate all unique pairs
    fit_pairs = list(combinations(this_mut_fits, 2))
    # calculate the difference for each pair and take the absolute value
    differences = [abs(i-j) for i,j in fit_pairs]
    # compute the mean of the differences
    mean_difference = np.mean(differences)
    fit_diffs[k] = mean_difference

# Move lag adv by some fraction of distance to mean:
lag_factor = 0.8

# Fluct pairs:    
pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]    

all_fluct_lags = []
all_fluct_errs = np.zeros([num_muts,20])
all_opp_fluct_errs = np.zeros([num_muts,20])
home_better=0
away_better=0
home_much_better=0
away_much_better=0
borderline=0
track_home_better = np.zeros([num_muts,20])
track_away_better = np.zeros([num_muts,20])
track_home_much_better = np.zeros([num_muts,20])
track_away_much_better = np.zeros([num_muts,20])
track_borderline = np.zeros([num_muts,20])
track_nonadd = np.zeros([num_muts,10])
track_mem = np.zeros([num_muts,10])
for k in range(num_muts):
    fluct_lags=[]
    fluct_errs=[]
    opp_fluct_errs=[]
    p=0
    for pair in pairs:
        mean_adv = np.mean(lag_advs[k])
        env1 = pair[0]; env2=pair[1]
        lag1 = lag_advs[k,env1]; lag2 = lag_advs[k,env2]
        fluct_lag1 = lag1 + lag_factor*(mean_adv-lag1)
        fluct_lag2 = lag2 + lag_factor*(mean_adv-lag2)
        fluct_lags.append([fluct_lag1,fluct_lag2])
        fluct_fit1 = r_anc[env1]*fluct_lag1
        fluct_fit2 = r_anc[env2]*fluct_lag2
        nonadd = np.abs((fluct_fit1 + fluct_fit2)/2 - (fitnesses[k,env1] + fitnesses[k,env2])/2)
        track_nonadd[k,p] = nonadd
        fluct_err1 = np.abs(fitnesses[k,env1]-fluct_fit1)
        fluct_err2 = np.abs(fitnesses[k,env2]-fluct_fit2)
        track_mem[k,p] = 0.5*(fluct_err1+fluct_err2)
        opp_fluct_err1 = np.abs(fitnesses[k,env2]-fluct_fit1)
        opp_fluct_err2 = np.abs(fitnesses[k,env1]-fluct_fit2)
        if fluct_err1 < opp_fluct_err1:
            home_better += 1
            track_home_better[k,2*p] = 1
        else:
            away_better += 1
            track_away_better[k,2*p] = 1
            
        if fluct_err2 < opp_fluct_err2:
            home_better += 1
            track_home_better[k,2*p+1] = 1
        else:
            away_better += 1
            track_away_better[k,2*p+1] = 1
            
            
        if fluct_err1 < opp_fluct_err1 - border_thr:
            home_much_better += 1
            track_home_much_better[k,2*p] = 1
        elif fluct_err1 > opp_fluct_err1 + border_thr:
            away_much_better += 1
            track_away_much_better[k,2*p] = 1
        else:
            borderline += 1
            track_borderline[k,2*p] = 1
            
        if fluct_err2 < opp_fluct_err2 - border_thr:
            home_much_better += 1
            track_home_much_better[k,2*p+1] = 1
        elif fluct_err2 > opp_fluct_err2 + border_thr:
            away_much_better += 1
            track_away_much_better[k,2*p+1] = 1
        else:
            borderline += 1
            track_borderline[k,2*p+1] = 1
        p+=1
        fluct_errs.extend([fluct_err1,fluct_err2])
        opp_fluct_errs.extend([opp_fluct_err1,opp_fluct_err2])
    all_fluct_errs[k] = fluct_errs
    all_opp_fluct_errs[k] = opp_fluct_errs


frac_home_better_model = home_better/(home_better+away_better)
frac_away_better_model = away_better/(home_better+away_better)
home_better_list = np.sum(track_home_better,1).tolist()
away_better_list = np.sum(track_away_better,1).tolist()

frac_home_much_better_model = home_much_better/(home_much_better+away_much_better+borderline)
frac_away_much_better_model = away_much_better/(home_much_better+away_much_better+borderline)
frac_borderline_model = borderline/(home_much_better+away_much_better+borderline)

home_much_better_list = np.sum(track_home_much_better,1).tolist()
away_much_better_list = np.sum(track_away_much_better,1).tolist()
borderline_list = np.sum(track_borderline,1).tolist()
data = {'Mean_Diff': fit_diffs.tolist(), 'Home_better_total': home_better_list, 'Away_better_total': away_better_list, 
        'Home_much_better_total': home_much_better_list, 'Away_much_better_total': away_much_better_list, 'Borderline_total': borderline_list,
        'Nonadditivity':np.mean(track_nonadd,1).tolist(), 'Memory':np.mean(track_mem,1).tolist()}
df_diff_err = pd.DataFrame(data)

median = df_diff_err['Mean_Diff'].quantile(0.9)
df_high = df_diff_err[df_diff_err['Mean_Diff'] > median]
plt.figure(figsize=(15,10))
frac_home_better_model_highdiff = np.sum(df_high.Home_better_total)/(np.sum(df_high.Home_better_total) + np.sum(df_high.Away_better_total))
frac_away_better_model_highdiff = np.sum(df_high.Away_better_total)/(np.sum(df_high.Home_better_total) + np.sum(df_high.Away_better_total))

frac_home_much_better_model_highdiff = np.sum(df_high.Home_much_better_total)/(np.sum(df_high.Home_much_better_total) + 
                                                                               np.sum(df_high.Away_much_better_total) +
                                                                               np.sum(df_high.Borderline_total))
frac_away_much_better_model_highdiff = np.sum(df_high.Away_much_better_total)/(np.sum(df_high.Home_much_better_total) + 
                                                                               np.sum(df_high.Away_much_better_total) +
                                                                               np.sum(df_high.Borderline_total))
frac_borderline_model_highdiff = np.sum(df_high.Borderline_total)/(np.sum(df_high.Home_much_better_total) + 
                                                                               np.sum(df_high.Away_much_better_total) +
                                                                               np.sum(df_high.Borderline_total))

median = df_diff_err['Mean_Diff'].quantile(0.1)
df_low = df_diff_err[df_diff_err['Mean_Diff'] < median]
plt.figure(figsize=(15,10))
frac_home_better_model_lowdiff = np.sum(df_low.Home_better_total)/(np.sum(df_low.Home_better_total) + np.sum(df_low.Away_better_total))
frac_away_better_model_lowdiff = np.sum(df_low.Away_better_total)/(np.sum(df_low.Home_better_total) + np.sum(df_low.Away_better_total))

frac_home_much_better_model_lowdiff = np.sum(df_low.Home_much_better_total)/(np.sum(df_low.Home_much_better_total) + 
                                                                             np.sum(df_low.Away_much_better_total) +
                                                                             np.sum(df_low.Borderline_total))
frac_away_much_better_model_lowdiff = np.sum(df_low.Away_much_better_total)/(np.sum(df_low.Home_much_better_total) + 
                                                                             np.sum(df_low.Away_much_better_total) +
                                                                             np.sum(df_low.Borderline_total))
frac_borderline_model_lowdiff = np.sum(df_low.Borderline_total)/(np.sum(df_low.Home_much_better_total) + 
                                                                             np.sum(df_low.Away_much_better_total) +
                                                                             np.sum(df_low.Borderline_total))
   
labels = ['All data','High fitness\ndifference','Low fitness\ndifference']
group1 = [frac_home_better_model,frac_home_better_model_highdiff,frac_home_better_model_lowdiff]
group2 = [frac_away_better_model,frac_away_better_model_highdiff,frac_away_better_model_lowdiff]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# 3 categories:
labels = ['All data','High fitness\ndifference','Low fitness\ndifference']
group1 = [frac_home_much_better_model,frac_home_much_better_model_highdiff,frac_home_much_better_model_lowdiff]
group2 = [frac_borderline_model,frac_borderline_model_highdiff,frac_borderline_model_lowdiff]
group3 = [frac_away_much_better_model,frac_away_much_better_model_highdiff,frac_away_much_better_model_lowdiff]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

plt.figure(figsize=(6,5))
fig, ax = plt.subplots(figsize=(6,5))
#bar1 = ax.bar(x - width/2, group1, width, label='Home', color = 'gray')
#bar2 = ax.bar(x + width/2, group2, width, label='Away', color = 'goldenrod')

#bar2 = ax.bar(x, group2, label='Other static environment', color='goldenrod')
#bar1 = ax.bar(x, group1, bottom=group2, label='Same static environment', color = 'gray')

bar3 = ax.bar(x, group3, label='Other static environment', color='goldenrod')
bar2 = ax.bar(x, group2, bottom=group3, label='Borderline', color='lightgray')
bar1 = ax.bar(x, group1, bottom=np.array(group2) + np.array(group3), label='Same static environment', color = 'lightblue')



# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel('Better predictor')
ax.set_ylabel('Fraction')
#ax.set_title('Scores by group')
ax.set_xticks(x)
ax.set_xticklabels(labels)
#ax.legend()
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim([0,1])

fig.tight_layout()
plt.show()