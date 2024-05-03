#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:15:55 2023

@author: clare
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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

get_fitness_path = '../../../software_module/callAtishFitness.py'
sys.path.append(os.path.dirname(os.path.expanduser(get_fitness_path)))
import callAtishFitness as inferFitness

sns.set_color_codes()
sns.set_style('white')
sns.set_style('ticks')

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#%% Import data:
    
BClist = pd.read_csv('../data/BClist.csv',index_col=0)
fitness_df = pd.read_csv('../data/fitness_df.csv',index_col=0)

#%% LEFT PANEL

plt.figure(figsize=(5.5,13))

plt.subplot(3,1,1)
plt.xlabel('Static env. 1 fitness')
plt.ylabel('Static env. 2 fitness')  
#plt.title(f'{conds[2]}')
plt.xlim(-0.8,0.8)
plt.ylim(-0.8,0.8)
xpoints = plt.xlim()
plt.plot(xpoints,xpoints,'k--') # identity line
plt.axvline(x=0,c='k')
plt.axhline(y=0,c='k')
#plt.xticks([])
#plt.yticks([])

plt.subplot(3,1,2)
plt.xlabel('Env. 1 fluct. - static fitness')
plt.ylabel('Env. 2 fluct. - static fitness')  
plt.xlim(-0.8,0.8)
plt.ylim(-0.8,0.8)
plt.axvline(x=0,c='k')
plt.axhline(y=0,c='k')
#plt.xticks([])
#plt.yticks([])
xpoints = np.linspace(*plt.xlim())
plt.plot(xpoints,-xpoints,'k-') # inverse identity line

plt.subplot(3,1,3)
plt.xlim(-0.8,1)
plt.ylim(-0.8,1)
plt.tight_layout()



#%% MAIN PANEL:
    
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

fluct_conds = [Gal_conds,L_conds,HN_conds]
low_lims = [-0.7,-0.6,-1.3]
high_lims = [0.7,.9,1.6]

pl=0
plt.figure(figsize=(15,13))
for conds in fluct_conds:#pair_conds:
    
    cond1=conds[0]
    cond2=conds[1]
    fluct_cond = conds[2]
    delta_cond1 = f'{cond1}_{fluct_cond}'
    delta_cond2 = f'{cond2}_{fluct_cond}'
    data = fitness_df[f'{cond1}_fitness']-fitness_df[f'{cond2}_fitness']
    delta1 = fitness_df[f'{delta_cond1}_fitness']-fitness_df[f'{cond1}_fitness']
    delta2 = fitness_df[f'{delta_cond2}_fitness']-fitness_df[f'{cond2}_fitness']
    vmin, vmax = np.min(data), np.max(data)
    
    plt.subplot(3,3,pl+1)
    plt.scatter(fitness_df[f'{cond1}_fitness'],fitness_df[f'{cond2}_fitness'],c=data,
                vmin=-max(abs(vmin), abs(vmax)), vmax=max(abs(vmin), abs(vmax)),
                cmap='seismic_r',alpha=0.5,edgecolors='gray',linewidths=0.5)
    if pl==0:
        plt.xlim(-0.6,0.8)
        plt.ylim(-0.6,0.8)
    elif pl==1:
        plt.xlim(-0.8,1)
        plt.ylim(-0.8,1)
    elif pl==2:
        plt.xlim(-1.9,2.6)
        plt.ylim(-1.9,2.6)
    ypoints = np.linspace(*plt.ylim())
    plt.plot(ypoints,ypoints,'k--') # identity line
    #plt.plot(xpoints,-xpoints,'k--') # inverse identity line
    plt.axvline(x=0,c='k')
    plt.axhline(y=0,c='k')
    plt.xlabel(f'Static {cond1} fitness')
    plt.ylabel(f'Static {cond2} fitness')
    
    plt.subplot(3,3,pl+4)
    plt.scatter(delta1,delta2,c=data,vmin=-max(abs(vmin), abs(vmax)), vmax=max(abs(vmin), abs(vmax)),
                cmap='seismic_r',alpha=0.5,edgecolors='gray',linewidths=0.5)
    if pl==0:
        plt.xlim(-1.1,0.5)
        plt.ylim(-1.1,1.5)
    elif pl==1:
        plt.xlim(-1.5,0.5)
        plt.ylim(-1.2,2.2)
    elif pl==2:
        plt.xlim(-2.5,1.8)
        plt.ylim(-0.5,1)
    xpoints = np.linspace(*plt.xlim())
    #plt.plot(xpoints,xpoints,'k--') # identity line
    plt.plot(xpoints,-xpoints,'k-') # inverse identity line
    plt.axvline(x=0,c='k')
    plt.axhline(y=0,c='k')
    plt.xlabel(f'{cond1} fluct. - static fitness')
    plt.ylabel(f'{cond2} fluct. - static fitness')
        
    plt.subplot(3,3,pl+7)
    plt.scatter(delta1,delta2,c=fitness_df['Mean_Diff_Static'],cmap='YlOrBr',alpha=0.9)
    if pl==0:
        plt.xlim(-1.1,0.5)
        plt.ylim(-1.1,1.5)
    elif pl==1:
        plt.xlim(-1.5,0.5)
        plt.ylim(-1.2,2.2)
    elif pl==2:
        plt.xlim(-2.5,1.8)
        plt.ylim(-0.5,1)
    xpoints = np.linspace(*plt.xlim())
    #plt.plot(xpoints,xpoints,'k--') # identity line
    plt.plot(xpoints,-xpoints,'k-') # inverse identity line
    plt.axvline(x=0,c='k')
    plt.axhline(y=0,c='k')
    plt.xlabel(f'{cond1} fluct. - static fitness')
    plt.ylabel(f'{cond2} fluct. - static fitness')
    
    pl+=1
    
plt.tight_layout()

    
#%% COLORBAR:
plt.scatter(delta1,delta2,c=fitness_df['Mean_Diff_Static'],cmap='YlOrBr',alpha=0.9)
plt.colorbar()

