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

#%% Figure 4A (data summary):
    
plt.figure(figsize=(5.2, 5))
# For Mean_Memory vs Mean_Diff_Static
x1 = fitness_df['Mean_Diff_Static'].values
y1 = fitness_df['Mean_Nonadditivity'].values
slope1 = np.linalg.lstsq(x1[:, np.newaxis], y1, rcond=None)[0][0]
line_x1 = np.linspace(np.min(x1), np.max(x1), 100)
line_y1 = slope1 * line_x1
plt.scatter(x1, y1, alpha=0.2, c='salmon')
plt.plot(line_x1, line_y1, 'salmon', label='Nonadditivity')

# For Mean_Change vs Mean_Diff_Static
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
    
#%% Figure 4B-C:
    
plt.figure(figsize=(15,10))

conds = [['Glu','Gal'],['Glu','Lac'],['Gal','Lac'],['Glu','H2O2'],['Glu','NaCl'],['H2O2','NaCl'],['Gal','H2O2'],
         ['Gal','NaCl'],['Lac','NaCl']]#,['Lac','H2O2']]

home_better=0
away_better=0
home_much_better=0
borderline=0
away_much_better=0
border_thr = 0.05

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
#border_thr = 0.04
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
#border_thr = 0.04
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
#bar1 = ax.bar(x - width/2, group1, width, label='Same static environment', color = 'gray')
#bar2 = ax.bar(x + width/2, group2, width, label='Other static environment', color= 'goldenrod')

bar3 = ax.bar(x, group3, label='Other static environment', color='goldenrod')
bar2 = ax.bar(x, group2, bottom=group3, label='Borderline', color='lightgray')
bar1 = ax.bar(x, group1, bottom=np.array(group2) + np.array(group3), label='Same static environment', color = 'lightblue')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Fraction')
#ax.set_title('Scores by group')
ax.set_xticks(x)
ax.set_xticklabels(labels)
#ax.legend()
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1),title='Better predictor of fluctuating fitness')
plt.ylim([0,1])
fig.tight_layout()
plt.savefig('Fig4B_nolegend_3cats.png',dpi=300)

# Controls:
labels = ['Within static\nreplicates','Within fluctuating\nreplicates']
group1 = [frac_home_much_better_staticreps,frac_home_much_better_fluctreps]
group2 = [frac_borderline_staticreps,frac_borderline_fluctreps]
group3 = [frac_away_much_better_staticreps,frac_away_much_better_fluctreps]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

plt.figure(figsize=(4.95,5))
fig, ax = plt.subplots(figsize=(4.95,5))
#bar1 = ax.bar(x - width/2, group1, width, label='Same environment',color='gray')
#bar2 = ax.bar(x + width/2, group2, width, label='Other environment',color='goldenrod')
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
plt.savefig('Fig4C_3cats.png',dpi=300)
plt.show()

