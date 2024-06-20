This folder contains code for inferring fitness of mutants based on raw frequency data, and formatting these fitness inferences for plotting in the figures.

The code was run in the following order:

# infer_fitness.py
This code uses the raw barcode frequency data (../../data/tidy_bc_fitness_counts_freqs.csv), to infer fitness of each barcoded mutant in each environment. 

The environments are either static (Glu, Gal, Lac, H, N) or fluctuating (all combinations of static), and the fitness is calculated for the two components of the fluctuating environments separately. The data in tidy_bc_fitness_counts_freqs.csv starts at timepoint 0, which is the end of the first growth cycle in which mutants, ancestors, and neutral strains were mixed. (Timepoint -1 is not included, which is the initial frequency after mixing but before growing together.)  Static environment measurements continued for five cycles, or up to timepoint 4. Fluctuating environment measurements continued for six cycles, or up to timepoint 5, to allow for equal time in both environmental components.

Static environment fitness inferences include timepoint 0, while fluctuating environment inferences typically do not and start at timepoint 1 instead. Since both inferences exclude timepoint -1, this means static inferences exclude frequency changes from the first growth cycle, while fluctuating inferences exclude the first two growth cycles. In two fluctuating environments (Glu/H and Lac/H), we included timepoint 0 and excluded timepoint 5, because the barcoded pool reached a frequency greater than 0.5 in the final timepoint (Supplementary Fig. 12). We excluded the final timepoint from the static environment H, for the same reason. These changes are denoted throughout this and the following codes as "exceptions," "rm_high_pool_frac," and "T0" or "not" ("T0" meaning timepoint 0 is included, "not" meaning it is not included).

The functions calc_fitness_MLE_naive and calc_fitness_MLE_naive_skip_zeros_static calculate fitness as described in the Methods section. These functions calculate a naive standard deviation by minimizing the negative log likelihood across all timepoints for a given mutant in a given condition. To generate a standard error, we calculated and saved simple estimates of fitness, without the minimization of the negative log likelihood, using the functions save_fitness and save_fitness_skip_zeros_static, for bootstrapping the minimization of the negative log likelihood with 250 trials in another piece of code (see subfolder fit_lists_for_bootstrap).

Also included in this code are the controls (shown in Supp. Fig. 8) for comparing memory in fluctuating environments to "memory" in static environments, and for confirming that memory is not an artifact of regression to the mean.

# format_data.py


# calc_fit_stats.py

# compose_fitness_df.py

