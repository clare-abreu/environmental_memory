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
from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.decomposition import PCA
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FixedLocator
from scipy.stats import gaussian_kde

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

index_dict = {}
index_dict['Glu/Gal'] = 'Glu-Gal'
index_dict['Glu/Lac'] = 'Glu-L'
index_dict['Glu/H2O2'] = 'high_H_Glu-H'
index_dict['Glu/NaCl'] = 'Glu-N'
index_dict['H2O2/NaCl'] = 'high_H_H-N'
index_dict['Gal/Lac'] = 'Gal-L'
index_dict['Gal/H2O2'] = 'high_H_Gal-H'
index_dict['Gal/NaCl'] = 'Gal-N'
index_dict['Lac/NaCl'] = 'L-N'
index_dict['Lac/H2O2'] = 'high_H_L-H'

#%% Import data:
    
BClist = pd.read_csv('../data/BClist.csv',index_col=0)
fitness_df = pd.read_csv('../data/fitness_df.csv',index_col=0)
freqs_df = pd.read_csv('../data/tidy_bc_fitness_counts_freqs.csv',index_col=0)
# Separate the neutral barcodes into their own dataframe (for calculating mean fitness later)
neutral_freqs_df = freqs_df[freqs_df['neutral']==True]

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
    start = np.mean([df[(df['condition']==f'{cond}{r}')&(df['time']==times[0])].loc[BC]['frequency'] for r in reps])
    aves.append(start)
    mean_slope = []
    for k in range(len(times)-1):
        this_time = times[k]
        next_time = times[k+1]
        slopes = [df[(df['condition']==f'{cond}{r}')&(df['time']==next_time)].loc[BC]['frequency'] - df[(df['condition']==f'{cond}{r}')&(df['time']==this_time)].loc[BC]['frequency'] for r in reps]
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
        #neut_change += np.mean(-np.log(df[f'{cond}-{r}-T{timept}']/df[f'{cond}-{r}-T{start_timept}']))
        neut_change += np.mean(-np.log(df[(df['condition']==f'{cond}{r}')&(df['time']==timept)]['frequency']/df[(df['condition']==f'{cond}{r}')&(df['time']==start_timept)]['frequency']))
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
    
def plot_three_conds_nine_BCs(df_combined_freq,df_neutral_freq,fluct_conds,all_BCs,all_BC_seqs,choose_colors=['k','k','k']):

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
            
            cond = index_dict[fluct_cond]
            if cond == 'Gal/Lac':
                reps = [1,2]
            else:
                reps = [1,2,3]
            k=1
            
            fluct_timepts = [1,2,3,4,5]
            
            plt.subplot(3,3,pl+1)
            ave_freqs = average_trajectories(df_combined_freq,cond,BC,fluct_timepts,reps)
            plt.plot([n+1 for n in fluct_timepts],ave_freqs,color='k')
            
            fits = []
            mean_fits=[]
            for k in range(len(fluct_timepts)-1):
                mean_fits.append(get_mean_fit(df_neutral_freq,cond,fluct_timepts[k+1],reps))
                env = fluct_pair[fluct_timepts[k+1]]
                fits.append(fitness_df[f'{env}_fitness'].loc[BC_seq])
            start_freq = ave_freqs[0]
            exp_traj = exp_trajectory(fluct_timepts,start_freq,fits,mean_fits)
            print(exp_traj)
            
            plt.plot([n+1 for n in fluct_timepts],exp_traj,'k--')
            
            plt.axvspan(2, 3, alpha=0.3, color=color_dict_gray[fluct_pair[0]])
            plt.axvspan(4, 5, alpha=0.3, color=color_dict_gray[fluct_pair[0]])
            plt.axvspan(3, 4, alpha=0.3, color=color_dict_gray[fluct_pair[1]])
            plt.axvspan(5, 6, alpha=0.3, color=color_dict_gray[fluct_pair[1]])
                
            plt.yscale('log')
                
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
 
            pl+=1
            c+=1

    plt.tight_layout()
    plt.show()
    

#%% Plot all:
fluct_conds = ['Glu/Gal','Glu/Lac','H2O2/NaCl']
all_BCs = [[1369,1114,1110],[516,586,548],[939,838,759]]  # different order than above but same mutants
BC_seqs_1 = [BClist.loc[BC,'Sequence'] for BC in all_BCs[0]]
BC_seqs_2 = [BClist.loc[BC,'Sequence'] for BC in all_BCs[1]]
BC_seqs_3 = [BClist.loc[BC,'Sequence'] for BC in all_BCs[2]]
all_BC_seqs = [BC_seqs_1,BC_seqs_2,BC_seqs_3]
choose_colors = ['lime','magenta','blue']
    
plot_three_conds_nine_BCs(freqs_df,neutral_freqs_df,fluct_conds,all_BCs,all_BC_seqs,choose_colors)

#%% Fig. 2E-H:
    
#%%
# PLOT NONADDITIVITY histogram
pair_conds = [['Glu','Gal'],['Glu','Lac'],['Gal','Lac'],['Glu','H2O2'],['Glu','NaCl'],['H2O2','NaCl'],['Gal','H2O2'],
         ['Gal','NaCl'],['Lac','NaCl']]#,['Lac','H2O2']]
pair_conds_drop_batch_fx = [['Glu','Gal'],['Glu','Lac'],['Glu','NaCl'],['H2O2','NaCl'],['Gal','H2O2'],
         ['Gal','NaCl'],['Lac','NaCl']]#,['Lac','H2O2']]

estimate_type = 'conservative'

if estimate_type == 'less_conservative':   
    pair_conds = pair_conds_drop_batch_fx

plt.figure(figsize=(10,5))
nonadd = []
noise = []
nonadd_95sig = []
nonadd_80sig = []
nonadd_67sig = []
Nonadd_mode_BCs = {}
Nonadd_mode_srcs = {}
Nonadd_Z_df = pd.DataFrame()
for conds in pair_conds:
        
    df = fitness_df.copy()
    df['Predicted Fluctuating Fitness'] = ""
    df['Predicted Fluctuating Fitness'] = df[f'{conds[0]}_fitness']*(.5)+df[f'{conds[1]}_fitness']*(.5)
    if estimate_type == 'conservative':
        # Conservative:
        df['Predicted Fluctuating Fitness Error'] = 0.5*np.sqrt((df[f'{conds[0]}_SEM_bootstrap'])**2+(df[f'{conds[1]}_SEM_bootstrap'])**2)#0.5*np.sqrt(df[f'{conds[0]}_SEM']**2+df[f'{conds[1]}_SEM']**2)
    elif estimate_type == 'less_conservative':   
        # Less conservative (SEM across reps):
        df['Predicted Fluctuating Fitness Error'] = 0.5*np.sqrt((df[f'{conds[0]}_SEM_across_reps'])**2+(df[f'{conds[1]}_SEM_across_reps'])**2)
        
    df['Nonadditivity'] = df[f'{conds[0]}/{conds[1]}_fitness'] - df['Predicted Fluctuating Fitness']
    
    if estimate_type == 'conservative':
        df['Nonadditivity Z score'] = np.abs(df[f'{conds[0]}/{conds[1]}_fitness'] - df['Predicted Fluctuating Fitness'])/\
                                        np.sqrt(df['Predicted Fluctuating Fitness Error']**2 + df[f'{conds[0]}/{conds[1]}_SEM_bootstrap']**2)
        Nonadd_Z_df[f'{conds[0]}/{conds[1]} Nonadditivity Z score'] = np.abs(df[f'{conds[0]}/{conds[1]}_fitness'] - df['Predicted Fluctuating Fitness'])/\
                                                            np.sqrt(df['Predicted Fluctuating Fitness Error']**2 + df[f'{conds[0]}/{conds[1]}_SEM_bootstrap']**2)
    elif estimate_type == 'less_conservative': 
        df['Nonadditivity Z score'] = np.abs(df[f'{conds[0]}/{conds[1]}_fitness'] - df['Predicted Fluctuating Fitness'])/\
                                        np.sqrt(df['Predicted Fluctuating Fitness Error']**2 + df[f'{conds[0]}/{conds[1]}_SEM_across_reps']**2)
        Nonadd_Z_df[f'{conds[0]}/{conds[1]} Nonadditivity Z score'] = np.abs(df[f'{conds[0]}/{conds[1]}_fitness'] - df['Predicted Fluctuating Fitness'])/\
                                                            np.sqrt(df['Predicted Fluctuating Fitness Error']**2 + df[f'{conds[0]}/{conds[1]}_SEM_across_reps']**2)
    
    nonadd_mode_BCs = df['Nonadditivity'][df['Nonadditivity']<-0.18].index.tolist()
    srcs = []
    for bc in nonadd_mode_BCs:
        srcs.append(fitness_df[fitness_df.index==bc]['Source_Env'].tolist()[0])
    Nonadd_mode_BCs[f'{conds[0]}/{conds[1]}'] = nonadd_mode_BCs
    Nonadd_mode_srcs[f'{conds[0]}/{conds[1]}'] = srcs
    
    for j in df.index:
        if not np.isnan(df.loc[j,'Nonadditivity']):
            if not np.isnan(df.loc[j,'Predicted Fluctuating Fitness Error']):
                nonadd.append(df.loc[j,'Nonadditivity'])
                err_sample = np.random.normal(df.loc[j,'Predicted Fluctuating Fitness'],df.loc[j,'Predicted Fluctuating Fitness Error'])
                noise_sample = err_sample-df.loc[j,'Predicted Fluctuating Fitness']
                noise.append(noise_sample)
                
                if df.loc[j,'Nonadditivity Z score'] > 1.96:
                    nonadd_95sig.append(1)
                else:
                    nonadd_95sig.append(0)
                    
                if df.loc[j,'Nonadditivity Z score'] > 1.28:
                    nonadd_80sig.append(1)
                else:
                    nonadd_80sig.append(0)
                    
                if df.loc[j,'Nonadditivity Z score'] > 0.44:
                    nonadd_67sig.append(1)
                else:
                    nonadd_67sig.append(0)


# Normalize:
    
# Compute histogram values for each dataset
hist_nonadd, bins_nonadd = np.histogram(nonadd, bins=200, density=True)
hist_noise, bins_noise = np.histogram(noise, bins=200, density=True)

# Normalize histogram values to max height of 1
hist_nonadd /= hist_nonadd.max()
hist_noise /= hist_noise.max()

# Estimate KDE and normalize
kde_nonadd = gaussian_kde(nonadd)
kde_noise = gaussian_kde(noise)

# Generate evenly spaced values for x-axis (from min to max of our data)
x_nonadd = np.linspace(np.min(nonadd), np.max(nonadd), 1000)
x_noise = np.linspace(np.min(noise), np.max(noise), 1000)

# Get KDE values for each x and normalize
kde_vals_nonadd = kde_nonadd(x_nonadd)
kde_vals_nonadd /= kde_vals_nonadd.max()

kde_vals_noise = kde_noise(x_noise)
kde_vals_noise /= kde_vals_noise.max()

# Plot the histograms
plt.figure(figsize=(6,2.5))

plt.bar(bins_nonadd[:-1], hist_nonadd, width=np.diff(bins_nonadd), color='darkorange', alpha=0.5)
plt.plot(x_nonadd, kde_vals_nonadd, color='darkorange', linewidth=3, label='Data')

plt.xlim(-1, 1)
plt.ylabel('Density')
plt.xlabel('Non-additivity (overall fitness - additive null)')
plt.tight_layout()
plt.show()


# Filter columns that contain 'Memory Z Score'
nonadd_z_score_cols = [col for col in Nonadd_Z_df.columns if 'Nonadditivity Z score' in col]

# Melt the DataFrame to long format for the selected columns
long_df = Nonadd_Z_df[nonadd_z_score_cols].melt()

# Plotting a single histogram
plt.figure(figsize=(6, 2.5))
sns.histplot(long_df['value'], color='gray', bins=125, kde=True)
plt.axvline(x=1.96, color='black', linestyle='-')
plt.xlim(0, 14)
#plt.title('Histogram of Memory Z Score Columns')
plt.xlabel('Non-additivity significance score')
plt.ylabel('')
plt.tight_layout()
plt.show()

nonaddsig = np.sum(nonadd_95sig)/len(nonadd_95sig)
print(f'Nonadditivity significant with 95% confidence in {100*nonaddsig}% of cases')

#%%
# PLOT MEMORY histogram
pair_conds = [['Glu','Gal'],['Glu','Lac'],['Gal','Lac'],['Glu','H2O2'],['Glu','NaCl'],['H2O2','NaCl'],['Gal','H2O2'],
         ['Gal','NaCl'],['Lac','NaCl']]#,['Lac','H2O2']]
pair_conds_drop_batch_fx = [['Glu','Gal'],['Glu','Lac'],['Glu','NaCl'],['H2O2','NaCl'],['Gal','H2O2'],
         ['Gal','NaCl'],['Lac','NaCl']]#,['Lac','H2O2']]
if estimate_type == 'less_conservative':
    pair_conds = pair_conds_drop_batch_fx

#plt.figure(figsize=(10,5))
mem = []
mem_noise = []
mem_95sig = []
mem_80sig = []
mem_67sig = []
Mem_Z_df = pd.DataFrame()
HiMem_BCs = {}
HiMem_srcs = {}
for conds in pair_conds:
    #fitness_df = adaptive_H_hybrid

    #fitness_df = adaptive_withoutT0_rm_high_pool_frac
            
    df = fitness_df.copy()
    for c in conds:
        df[f'{c} Memory'] = df[f'{c}_{conds[0]}/{conds[1]}_fitness'] - df[f'{c}_fitness']
        
        
        if estimate_type == 'conservative':
            # Conservative:
            df[f'{c} Memory Z score'] = np.abs(df[f'{c}_{conds[0]}/{conds[1]}_fitness'] - df[f'{c}_fitness'])/\
                                            np.sqrt(df[f'{c}_{conds[0]}/{conds[1]}_SEM_bootstrap']**2 + df[f'{c}_SEM_bootstrap']**2)
            Mem_Z_df[f'{c}_{conds[0]}/{conds[1]} Memory Z score'] = np.abs(df[f'{c}_{conds[0]}/{conds[1]}_fitness'] - df[f'{c}_fitness'])/\
                                        np.sqrt(df[f'{c}_{conds[0]}/{conds[1]}_SEM_bootstrap']**2 + df[f'{c}_SEM_bootstrap']**2)
        if estimate_type == 'less_conservative':
            # Less conservative:
            df[f'{c} Memory Z score'] = np.abs(df[f'{c}_{conds[0]}/{conds[1]}_fitness'] - df[f'{c}_fitness'])/\
                                            np.sqrt(df[f'{c}_{conds[0]}/{conds[1]}_SEM_across_reps']**2 + df[f'{c}_SEM_across_reps']**2)
            Mem_Z_df[f'{c}_{conds[0]}/{conds[1]} Memory Z score'] = np.abs(df[f'{c}_{conds[0]}/{conds[1]}_fitness'] - df[f'{c}_fitness'])/\
                                        np.sqrt(df[f'{c}_{conds[0]}/{conds[1]}_SEM_across_reps']**2 + df[f'{c}_SEM_across_reps']**2)
        
        hi_memBCs = df[f'{c} Memory'][np.abs(df[f'{c} Memory'])>0.25].index.tolist()
        srcs = []
        for bc in hi_memBCs:
            srcs.append(fitness_df[fitness_df.index==bc]['Source_Env'].tolist()[0])
        HiMem_BCs[f'{c}_{conds[0]}/{conds[1]}'] = hi_memBCs
        HiMem_srcs[f'{c}_{conds[0]}/{conds[1]}'] = srcs
    
    for j in df.index:
        for c in conds:
            if not np.isnan(df.loc[j,f'{c} Memory']):
                if not np.isnan(df.loc[j,f'{c}_SEM_bootstrap']):
                    mem.append(df.loc[j,f'{c} Memory'])
                    if estimate_type == 'conservative':
                        # Conservative:
                        mem_err_sample = np.random.normal(df.loc[j,f'{c}_fitness'],df.loc[j,f'{c}_SEM_bootstrap'])
                    elif estimate_type == 'less_conservative':
                        # Less conservative:
                        mem_err_sample = np.random.normal(df.loc[j,f'{c}_fitness'],df.loc[j,f'{c}_SEM_across_reps'])
                    mem_noise_sample = mem_err_sample-df.loc[j,f'{c}_fitness']
                    mem_noise.append(mem_noise_sample)
                    
                    if df.loc[j,f'{c} Memory Z score'] > 1.96:
                        mem_95sig.append(1)
                    else:
                        mem_95sig.append(0)
                        
                    if df.loc[j,f'{c} Memory Z score'] > 1.28:
                        mem_80sig.append(1)
                    else:
                        mem_80sig.append(0)
                        
                    if df.loc[j,f'{c} Memory Z score'] > 0.44:
                        mem_67sig.append(1)
                    else:
                        mem_67sig.append(0)

# Normalize:
    
# Compute histogram values for each dataset
hist_mem, bins_mem = np.histogram(mem, bins=400, density=True)
hist_mem_noise, bins_mem_noise = np.histogram(mem_noise, bins=400, density=True)

# Normalize histogram values to max height of 1
hist_mem /= hist_mem.max()
hist_mem_noise /= hist_mem_noise.max()

# Estimate KDE and normalize
kde_mem = gaussian_kde(mem)
kde_mem_noise = gaussian_kde(mem_noise)

# Generate evenly spaced values for x-axis (from min to max of our data)
x_mem = np.linspace(np.min(mem), np.max(mem), 1000)
x_mem_noise = np.linspace(np.min(mem_noise), np.max(mem_noise), 1000)

# Get KDE values for each x and normalize
kde_vals_mem = kde_mem(x_mem)
kde_vals_mem /= kde_vals_mem.max()

kde_vals_mem_noise = kde_mem_noise(x_mem_noise)
kde_vals_mem_noise /= kde_vals_mem_noise.max()

# Plot the histograms
plt.figure(figsize=(6,2.5))

plt.bar(bins_mem[:-1], hist_mem, width=np.diff(bins_mem), color='darkorange', alpha=0.5)
plt.plot(x_mem, kde_vals_mem, color='darkorange', linewidth=3, label='Data')

plt.xlim(-1, 1)
plt.ylabel('Density')
plt.xlabel('Memory (fluct. component fitness - static null)')
plt.tight_layout()
plt.show()


# Filter columns that contain 'Memory Z Score'
memory_z_score_cols = [col for col in Mem_Z_df.columns if 'Memory Z score' in col]

# Melt the DataFrame to long format for the selected columns
long_df = Mem_Z_df[memory_z_score_cols].melt()

# Plotting a single histogram
plt.figure(figsize=(6, 2.5))
sns.histplot(long_df['value'], color='gray', bins=250, kde=True)
plt.axvline(x=1.96, color='black', linestyle='-')
plt.xlim(0, 14)
#plt.title('Histogram of Memory Z Score Columns')
plt.xlabel('Memory significance score')
plt.ylabel('')
plt.tight_layout()
plt.show()


memsig = np.sum(mem_95sig)/len(mem_95sig)
print(f'Memory significant with 95% confidence in {100*memsig}% of cases')

