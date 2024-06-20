This folder contains code for inferring fitness of mutants based on raw frequency data, and formatting these fitness inferences for plotting in the figures.

The code was run in the following order:

# infer_fitness.py
This code uses the raw barcode frequency data (../../data/tidy_bc_fitness_counts_freqs.csv), to infer fitness of each barcoded mutant in each environment. 

The environments are either static (Glu, Gal, Lac, H, N) or fluctuating (all combinations of static), and the fitness is calculated for the two components of the fluctuating environments separately. The data in tidy_bc_fitness_counts_freqs.csv starts at timepoint 0, which is the end of the first growth cycle in which mutants, ancestors, and neutral strains were mixed. (Timepoint -1 is not included, which is the initial frequency after mixing but before growing together.)  Static environment measurements continued for five cycles, or up to timepoint 4. Fluctuating environment measurements continued for six cycles, or up to timepoint 5, to allow for equal time in both environmental components.

Static environment fitness inferences include timepoint 0, while fluctuating environment inferences typically do not and start at timepoint 1 instead. This means that the static inferences



# format_data.py

# calc_fit_stats.py

# compose_fitness_df.py

