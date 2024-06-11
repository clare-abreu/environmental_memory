#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:42:36 2024

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
import ast

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
# High-variance mutants:
df_high_rep_variation_mutants = pd.read_csv('../data/high_rep_variation_mutants.csv',index_col=0)
high_rep_variation_mutants = df_high_rep_variation_mutants.index.tolist()
fitness_df_dropped = fitness_df.drop(high_rep_variation_mutants)
# COMMENT OUT IF NOT DROPPING HIGH VARIANCE MUTANTS:
fitness_df = fitness_df_dropped

BC_throw_out = pd.read_csv('../data/WGS/BCs_to_throw_out.csv',index_col=0)
reseq_samples_throw_out = BC_throw_out.index.tolist()
reseq_BCs = pd.read_csv('../data/WGS/WGS_ReSeq_indices_BC_map_Source.csv',index_col=0)
samples_throw_out = reseq_BCs.loc[reseq_samples_throw_out,'oldIndex'].tolist()
admeraIndices = pd.read_csv('../data/WGS/admeraIndices.csv',index_col=0)
BC_map = pd.read_csv('../data/WGS/WGS_indices_BC_map.csv',index_col=0)
shaili_ploidy_assay = pd.read_csv('../data/WGS/SM_benomyl_assay_df.csv')

WGS_homo_snps = pd.read_csv('../data/WGS/WGS_consolidated_homo_SNPs_by_sample.csv',index_col=0)
WGS_hetero_snps = pd.read_csv('../data/WGS/WGS_consolidated_hetero_SNPs_by_sample.csv',index_col=0)

Reseq_map = pd.read_csv('../data/WGS/WGS_ReSeq_indices_BC_map.csv',index_col=0)
CNV_notes = pd.read_csv('../data/WGS/CNV_notes.csv',index_col=0)


#%%

# Add sample names to ploidy assay df:
shaili_ploidy_assay['sample'] = ""
samples=[]
for k in range(len(shaili_ploidy_assay)):
    row = shaili_ploidy_assay.iloc[k]['row']
    col = shaili_ploidy_assay.iloc[k]['column']
    plate = shaili_ploidy_assay.iloc[k]['plate']
    if col < 10:
        well = f'{row}0{col}'
    else:
        well = f'{row}{col}'
        
    admeraIndex = admeraIndices.loc[well,'admeraIndex']
    if admeraIndex < 10:
        sample = f'WGS-March23-143-0{plate}-0{admeraIndex}'
    else:
        sample = f'WGS-March23-143-0{plate}-{admeraIndex}'
        
    samples.append(sample)

shaili_ploidy_assay['sample'] = samples

# Map sample names to barcodes:
shaili_ploidy_assay['BC'] = ""
BCs = []
for k in range(len(shaili_ploidy_assay)):
    sample = shaili_ploidy_assay.iloc[k]['sample']
    BC = BC_map.loc[sample,'BC']
    BCs.append(BC)
shaili_ploidy_assay['BC'] = BCs
shaili_ploidy_assay.set_index('BC',inplace=True)

# Map sample names to mean fit diff, mean change, mean memory
shaili_ploidy_assay['Source_Env'] = ""
shaili_ploidy_assay['Mean_Diff_Static'] = np.nan
shaili_ploidy_assay['Mean_Memory'] = np.nan
shaili_ploidy_assay['Mean_Nonadditivity'] = np.nan

shaili_ploidy_assay = shaili_ploidy_assay.drop(['notes','doublechecked'], axis=1)

for bc in shaili_ploidy_assay.index:
    for col in ['Source_Env','Mean_Diff_Static','Mean_Memory','Mean_Nonadditivity']:
        if bc in fitness_df.index:
            shaili_ploidy_assay.loc[bc,col] = fitness_df.loc[bc,col]

# Drop out barcodes that we threw out due to multiple BC reads, and Alice/Bob barcodes (no stats in fitness_df):
list_to_exclude = samples_throw_out
shaili_ploidy_filtered = shaili_ploidy_assay[~shaili_ploidy_assay['sample'].isin(list_to_exclude)]
shaili_ploidy_filtered = shaili_ploidy_filtered[shaili_ploidy_filtered['Source_Env'] != '']

# Separate diploids:

diploids = shaili_ploidy_filtered[shaili_ploidy_filtered['growth inhibited (diploid)']==True]
diploids_minus_borderline = diploids[diploids['borderline']!=True]
diploids_plus_borderline = shaili_ploidy_filtered[(shaili_ploidy_filtered['growth inhibited (diploid)']==True) | 
                                                  (shaili_ploidy_filtered['borderline']==True)]
haploids = shaili_ploidy_filtered[shaili_ploidy_filtered['growth inhibited (diploid)']==False]


# %% Pure diploid search

Reseq_map['Source_Env'] = ""
for k in Reseq_map.index:
    this_BC = Reseq_map.loc[k,'BC']
    if this_BC in BClist.index:
        Reseq_map.loc[k,'Source_Env'] = BClist.loc[this_BC,'Source']


def parse_list(lst_str):
    try:
        return ast.literal_eval(lst_str)
    except (SyntaxError, ValueError):
        return []

WGS_homo_snps['mutations'] = WGS_homo_snps['mutations'].apply(parse_list)
WGS_hetero_snps['mutations'] = WGS_hetero_snps['mutations'].apply(parse_list)

def filter_synonymous(lst):
    return [item for item in lst if 'synonymous' not in item]

def filter_None(lst):
    return [item for item in lst if 'None' not in item]

WGS_homo_snps['mutations'] = WGS_homo_snps['mutations'].apply(lambda x: filter_synonymous(x))
WGS_hetero_snps['mutations'] = WGS_hetero_snps['mutations'].apply(lambda x: filter_synonymous(x))

WGS_homo_snps['mutations'] = WGS_homo_snps['mutations'].apply(lambda x: filter_None(x))
WGS_hetero_snps['mutations'] = WGS_hetero_snps['mutations'].apply(lambda x: filter_None(x))

# Find samples with no mutations:
WGS_homo_muts=[]
WGS_hetero_muts=[]
for s in shaili_ploidy_assay['sample'].tolist():
    if s in WGS_homo_snps.index:
        if WGS_homo_snps.loc[s,'mutations'] != []:
            WGS_homo_muts.append(s)
    if s in WGS_hetero_snps.index:
        if WGS_hetero_snps.loc[s,'mutations'] != []:
            WGS_hetero_muts.append(s)


all_muts = np.unique(WGS_homo_muts.extend(WGS_hetero_muts))

pure_diploids = diploids[~diploids['sample'].isin(all_muts)]

#%% Iron pathway SNP mutations

def contains_substring(lst, substring):
    return any(substring in str for str in lst)

substring = 'AFT'
WGS_homo_AFT = WGS_homo_snps[WGS_homo_snps['mutations'].apply(lambda x: contains_substring(x, substring))]
WGS_hetero_AFT = WGS_hetero_snps[WGS_hetero_snps['mutations'].apply(lambda x: contains_substring(x, substring))]

AFT_muts=[]
AFT_muts.extend(WGS_homo_AFT.index)
AFT_muts.extend(WGS_hetero_AFT.index)

AFT_muts = np.unique(AFT_muts).tolist()
AFT_BCs = []
for sample in AFT_muts:
    BC = BC_map.loc[sample,'BC']
    AFT_BCs.append(BC)

keep_AFT_BCs=[]
for BC in AFT_BCs:
    if BC in fitness_df.index:
        keep_AFT_BCs.append(BC)
AFT_fitness_df = fitness_df.loc[keep_AFT_BCs]

substring = 'YGL220W'
WGS_homo_BOL = WGS_homo_snps[WGS_homo_snps['mutations'].apply(lambda x: contains_substring(x, substring))]
WGS_hetero_BOL = WGS_hetero_snps[WGS_hetero_snps['mutations'].apply(lambda x: contains_substring(x, substring))]

BOL_muts=[]
BOL_muts.extend(WGS_homo_BOL.index)
BOL_muts.extend(WGS_hetero_BOL.index)

BOL_muts = np.unique(BOL_muts).tolist()
BOL_BCs = []
for sample in BOL_muts:
    BC = BC_map.loc[sample,'BC']
    BOL_BCs.append(BC)

keep_BOL_BCs=[]
for BC in BOL_BCs:
    if BC in fitness_df.index:
        keep_BOL_BCs.append(BC)
BOL_fitness_df = fitness_df.loc[keep_BOL_BCs]

AFT_BOL_fitness_df = fitness_df.loc[keep_AFT_BCs+keep_BOL_BCs]

#%% Salt pump CNV mutations

# CNVs:

samples = np.unique(CNV_notes.index.tolist())
BCs = BC_map.loc[samples].BC.tolist()
BCs_in_index = [BC for BC in BCs if BC in fitness_df.index]

ENA_CNV = CNV_notes[CNV_notes['Gene'].str.contains('ENA', na=False)]

overlap_BCs = list(set(reseq_BCs.oldIndex.tolist()) & set(ENA_CNV.index.tolist()))
drop_BCs = samples_throw_out + [x for x in overlap_BCs if x not in ['WGS-March23-143-04-88',
                                     'WGS-March23-143-01-45',
                                     'WGS-March23-143-03-56']]
ENA_CNV = ENA_CNV.drop(drop_BCs,errors='ignore')

ENA_samples = np.unique(ENA_CNV.index.tolist())
ENA_BCs = BC_map.loc[ENA_samples].BC.tolist()
ENA_BCs_in_index = [BC for BC in ENA_BCs if BC in fitness_df.index]

ENA_samples_slight_amp = np.unique(ENA_CNV[ENA_CNV['Notes'].str.contains('slightly', na=False)].index.tolist())
ENA_slight_amp_BCs = BC_map.loc[ENA_samples_slight_amp].BC.tolist()
ENA_slight_amp_BCs_in_index = [BC for BC in ENA_slight_amp_BCs if BC in fitness_df.index]

ENA_samples_amp = np.unique(ENA_CNV[ENA_CNV['Notes'].str.contains('especially', na=False)].index.tolist())
ENA_amp_BCs = BC_map.loc[ENA_samples_amp].BC.tolist()
ENA_amp_BCs_in_index = [BC for BC in ENA_amp_BCs if BC in fitness_df.index]

ENA_not_amp_BCs_in_index = [item for item in ENA_BCs_in_index if item not in ENA_amp_BCs_in_index]

# SNPs:
    
substring = 'ENA'
WGS_homo_ENA = WGS_homo_snps[WGS_homo_snps['mutations'].apply(lambda x: contains_substring(x, substring))]
WGS_hetero_ENA = WGS_hetero_snps[WGS_hetero_snps['mutations'].apply(lambda x: contains_substring(x, substring))]

ENA_muts=[]
ENA_muts.extend(WGS_homo_ENA.index)
ENA_muts.extend(WGS_hetero_ENA.index)

ENA_muts = np.unique(ENA_muts).tolist()
ENA_BCs = []
for sample in ENA_muts:
    BC = BC_map.loc[sample,'BC']
    ENA_BCs.append(BC)

keep_ENA_BCs=[]
for BC in ENA_BCs:
    if BC in fitness_df.index:
        keep_ENA_BCs.append(BC)
ENA_fitness_df = fitness_df.loc[keep_ENA_BCs]



#%% Diploids without iron or salt mutations
to_drop = ENA_BCs_in_index + keep_AFT_BCs + keep_BOL_BCs + keep_ENA_BCs
to_drop = [index for index in to_drop if index in diploids_minus_borderline.index]
diploids_minus_borderline_minus_muts = diploids_minus_borderline.drop(to_drop)

#%% Total comparison plot:
    
plt.figure(figsize=(5.2,5))
AFT_BOL_mean_diff = np.mean(AFT_BOL_fitness_df['Mean_Diff_Static'])
AFT_BOL_SEM_diff = np.std(AFT_BOL_fitness_df['Mean_Diff_Static'])/np.sqrt(len(AFT_BOL_fitness_df['Mean_Diff_Static']))
AFT_BOL_mean_change = np.mean(AFT_BOL_fitness_df['Mean_Memory'])
AFT_BOL_SEM_change = np.std(AFT_BOL_fitness_df['Mean_Memory'])/np.sqrt(len(AFT_BOL_fitness_df['Mean_Memory']))

ENA_SNP_mean_diff = np.mean(ENA_fitness_df['Mean_Diff_Static'])
ENA_SNP_SEM_diff = np.std(ENA_fitness_df['Mean_Diff_Static'])/np.sqrt(len(ENA_fitness_df['Mean_Diff_Static']))
ENA_SNP_mean_change = np.mean(ENA_fitness_df['Mean_Memory'])
ENA_SNP_SEM_change = np.std(ENA_fitness_df['Mean_Memory'])/np.sqrt(len(ENA_fitness_df['Mean_Memory']))

ENA_not_amp_mean_diff = np.mean(fitness_df.loc[ENA_not_amp_BCs_in_index].Mean_Diff_Static)
ENA_not_amp_SEM_diff = np.std(fitness_df.loc[ENA_not_amp_BCs_in_index].Mean_Diff_Static)/np.sqrt(len(fitness_df.loc[ENA_not_amp_BCs_in_index].Mean_Diff_Static))
ENA_not_amp_mean_change = np.mean(fitness_df.loc[ENA_not_amp_BCs_in_index].Mean_Memory)
ENA_not_amp_SEM_change = np.std(fitness_df.loc[ENA_not_amp_BCs_in_index].Mean_Memory)/np.sqrt(len(fitness_df.loc[ENA_not_amp_BCs_in_index].Mean_Memory))

ENA_mean_diff = np.mean(fitness_df.loc[ENA_BCs_in_index].Mean_Diff_Static)
ENA_SEM_diff = np.std(fitness_df.loc[ENA_BCs_in_index].Mean_Diff_Static)/np.sqrt(len(fitness_df.loc[ENA_BCs_in_index].Mean_Diff_Static))
ENA_mean_change = np.mean(fitness_df.loc[ENA_BCs_in_index].Mean_Memory)
ENA_SEM_change = np.std(fitness_df.loc[ENA_BCs_in_index].Mean_Memory)/np.sqrt(len(fitness_df.loc[ENA_BCs_in_index].Mean_Memory))

ENA_amp_mean_diff = np.mean(fitness_df.loc[ENA_amp_BCs_in_index].Mean_Diff_Static)
ENA_amp_SEM_diff = np.std(fitness_df.loc[ENA_amp_BCs_in_index].Mean_Diff_Static)/np.sqrt(len(fitness_df.loc[ENA_amp_BCs_in_index].Mean_Diff_Static))
ENA_amp_mean_change = np.mean(fitness_df.loc[ENA_amp_BCs_in_index].Mean_Memory)
ENA_amp_SEM_change = np.std(fitness_df.loc[ENA_amp_BCs_in_index].Mean_Memory)/np.sqrt(len(fitness_df.loc[ENA_amp_BCs_in_index].Mean_Memory))

dip_mean_diff = np.mean(diploids_minus_borderline_minus_muts['Mean_Diff_Static'])
dip_SEM_diff = np.std(diploids_minus_borderline_minus_muts['Mean_Diff_Static'])/np.sqrt(len(diploids_minus_borderline_minus_muts['Mean_Diff_Static']))
dip_mean_change = np.mean(diploids_minus_borderline_minus_muts['Mean_Memory'])
dip_SEM_change = np.std(diploids_minus_borderline_minus_muts['Mean_Memory'])/np.sqrt(len(diploids_minus_borderline_minus_muts['Mean_Memory']))

#plt.errorbar([dip_mean_diff,AFT_BOL_mean_diff,ENA_amp_mean_diff],[dip_mean_change,AFT_BOL_mean_change,ENA_amp_mean_change],
#             xerr=[dip_SEM_diff,AFT_BOL_SEM_diff,ENA_amp_SEM_diff],yerr=[dip_SEM_change,AFT_BOL_SEM_change,ENA_amp_SEM_change])

colors = ['peru', 'magenta', 'blue']  # define the colors for the points
labels = ['Diploids', 'Iron pathway (AFT/BOL)', 'Sodium pump (ENA)']  # define the labels for the points

x_values = [dip_mean_diff, AFT_BOL_mean_diff, ENA_mean_diff]
y_values = [dip_mean_change, AFT_BOL_mean_change, ENA_mean_change]
x_errors = [dip_SEM_diff, AFT_BOL_SEM_diff, ENA_SEM_diff]
y_errors = [dip_SEM_change, AFT_BOL_SEM_change, ENA_SEM_change]

for x, y, xerr, yerr, color, label in zip(x_values, y_values, x_errors, y_errors, colors, labels):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=color, label=label)
# All Mean_Memory vs Mean_Diff_Static
x2 = fitness_df['Mean_Diff_Static'].values
y2 = fitness_df['Mean_Memory'].values
x_dip = diploids_minus_borderline_minus_muts['Mean_Diff_Static'].values
y_dip = diploids_minus_borderline_minus_muts['Mean_Memory'].values
x_aft = AFT_BOL_fitness_df['Mean_Diff_Static'].values
y_aft = AFT_BOL_fitness_df['Mean_Memory'].values
x_ena = fitness_df.loc[ENA_BCs_in_index].Mean_Diff_Static.values
y_ena = fitness_df.loc[ENA_BCs_in_index].Mean_Memory.values
slope2 = np.linalg.lstsq(x2[:, np.newaxis], y2, rcond=None)[0][0]
line_x2 = np.linspace(np.min(x2), np.max(x2), 100)
line_y2 = slope2 * line_x2
plt.scatter(x2, y2, alpha=0.1, c='gray')#'steelblue')
plt.scatter(x_dip, y_dip, alpha=0.2, c='peru')#'steelblue')
plt.scatter(x_aft, y_aft, alpha=0.2, c='magenta')
plt.scatter(x_ena, y_ena, alpha=0.2, c='blue')

plt.xlabel('Mean fitness difference (static)')
plt.ylabel('Mean fitness change (fluctuating)')

plt.xlabel('Mean fitness difference (static)')
plt.ylabel('Mean memory (fluctuating)')
#plt.legend()  # show a legend with the labels
plt.tight_layout()
plt.show()

plt.figure(figsize=(5.2,5))
#plt.errorbar([ENA_not_amp_mean_diff, ENA_SNP_mean_diff, ENA_amp_mean_diff], [ENA_not_amp_mean_change, ENA_SNP_mean_change, ENA_amp_mean_change], 
#             xerr=[ENA_not_amp_SEM_diff, ENA_SNP_SEM_diff, ENA_amp_SEM_diff], yerr=[ENA_not_amp_SEM_change, ENA_SNP_SEM_change, ENA_amp_SEM_change], fmt='o', color='blue')
plt.errorbar(ENA_not_amp_mean_diff, ENA_not_amp_mean_change, 
             xerr=ENA_not_amp_SEM_diff, yerr=ENA_not_amp_SEM_change, 
             fmt='o', color='cornflowerblue')
plt.errorbar(ENA_SNP_mean_diff, ENA_SNP_mean_change, 
             xerr=ENA_SNP_SEM_diff, yerr=ENA_SNP_SEM_change, 
             fmt='o', color='gray')
plt.errorbar(ENA_amp_mean_diff, ENA_amp_mean_change, 
             xerr=ENA_amp_SEM_diff, yerr=ENA_amp_SEM_change, 
             fmt='o', color='navy')
plt.xlabel('Mean fitness difference (static)')
plt.ylabel('Mean memory (fluctuating)')
# All Mean_Memory vs Mean_Diff_Static
x2 = fitness_df['Mean_Diff_Static'].values
y2 = fitness_df['Mean_Memory'].values
x_lowcn = fitness_df.loc[ENA_not_amp_BCs_in_index]['Mean_Diff_Static'].values
y_lowcn = fitness_df.loc[ENA_not_amp_BCs_in_index]['Mean_Memory'].values
x_hicn = fitness_df.loc[ENA_amp_BCs_in_index]['Mean_Diff_Static'].values
y_hicn = fitness_df.loc[ENA_amp_BCs_in_index]['Mean_Memory'].values
slope2 = np.linalg.lstsq(x2[:, np.newaxis], y2, rcond=None)[0][0]
line_x2 = np.linspace(np.min(x2), np.max(x2), 100)
line_y2 = slope2 * line_x2
plt.scatter(x2, y2, alpha=0.1, c='gray')#'steelblue')
plt.scatter(x_lowcn, y_lowcn, alpha=0.2, c='cornflowerblue')#'steelblue')
plt.scatter(x_hicn, y_hicn, alpha=0.2, c='navy')#'steelblue')

#plt.legend()  # show a legend with the labels
plt.tight_layout()
plt.show()
