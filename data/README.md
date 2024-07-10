This folder contains the data used in used in Abreu et al (Nature Ecology & Evolution, 2024).

The subfolder Evo_counts contains barcode sequencing data from evolution experiments, and the subfolder WGS contains data from whole-genome sequencing.

The files included here are:

# BClist.csv
This table contains basic information about the barcode pool.

COLUMNS:

BCID: A barcode ID we sometimes use to identify the mutants.

Sequence: DNA sequences of barcodes.

Source: Evolution environment in which mutants evolved. 'DPY256' indicates the ancestral strain (which contains a restriction enzyme that we digested-- it therefore does not appear at actual frequency in measurement data), and 'Venkataram' indicates barcoded strains with neutral (ancestral) fitness used to infer mean population fitness during measurements. The rest are the mutants, drawn from two evolution experiments labeled 'Circ' and 'SEvo.' 'A' and 'B' are mutants being analyzed in a separate upcoming study. 'G': glucose; 'H': H2O2; 'N': NaCl; 'L': lactate; 'Gal': galactose; 'HF': fluctuating G/H; 'LF': fluctuating G/L; 'GalF': fluctuating Gal/G; 'GN' or 'NF': fluctuating G/N; 'HN': fluctuating H/N.

Color: A color palette than can be used to differentiate the mutant sources in plots.

# fitness_df.csv
This table contains fitness inferences of the set of mutants deemed adaptive in their home evolution environment, and calculated as described in code/infer_fitness.

Columns denote fitness in a a particular environment as {environment}_fitness. Fitness in fluctuating environments is given for individual components (i.e. {component1}_{component1/component2}_fitness for the fitness in the first component of the environment), and also overall as an average of the two components (and denoted {component1/component2}_fitness). Absolute values of fitness differences across pairs of environments are given (i.e. {component1/component2}_diff, as well as the mean difference of all 10 pairs of static environments (Mean_Diff_Static). Mean_Origin_Distance_Static and Mean_Sine_Angle_Diff_Static are defined in Supp. Fig. 10 and analyzed in code/angle_origin_distance.Rmd. Mean_Nonadditivity and Mean_Memory are defined in the main text, and averaged across all fluctuating environments except for Lac/H2O2, as explained in the main text and Supp. Fig. 2. Standard errors (SEM) are calculated using two definitions: across replicates (standard deviation of the three replicate inferences, divided by sqrt(3)), and with a bootstrap (sampling with replacement from all timepoints from all replicates and minimizing negative log likelihood, as explained in infer_fitness). The latter is a more conservative estimate of error. (See Supp. Fig. 6 for a comparison of more/less conservative estimates.)

# high_rep_variation_mutants.csv
This table lists 12 mutants with mean replicate-to-replicate fitness difference greater than 0.2. We exclude these mutants from overall fitness statistics, but include them in individual analyses.

# tidy_bc_fitness_counts_freqs.csv
This table lists raw frequency counts of barcoded mutants from fitness assays.

COLUMNS:

BCID: A barcode ID we sometimes use to identify the mutants.

barcode: DNA sequences of barcodes.

condition: Environment where fitness is being measured (followed by a number indicating the replicate). Static environments are listed as Glu, Gal, L, H, or N. H (H2O2) measurements were repeated once we learned that concentration decreased over time if H2O2 was added to media in weeks prior to experiments (see Methods). In the repeated assays, we added H2O2 directly at the transfer/dilution step; these conditions are labeled 'high_H' and are represented in the main text data rather than the simple 'H' measurements. Fluctuating environments are denoted with hyphens. The conditions 'A', 'B', 'GGL', 'GLG', and 'PP' are data analyzed in an upcoming study.

count: Counts of barcodes from sequencing data.

time: Timepoints during assays, which go from 0 to 5 in fluctuating environments and 0 to 4 in static environments. See code/infer_fitness for a discussion of when some timepoints are excluded from inferences.

neutral: A true/false classification of barcoded neutral strains (with ancestral fitness) used to calculate mean population fitness (see Methods).

marked_neutral: Same as the "neutral" column, but includes six strains that were found to be adaptive in Gal. We exclude these strains from mean population fitness calculations.

frequency: frequency of each barcode lineage in a given timepoint/condition, or the count divided by the sum of counts in that timepoint/condition.

