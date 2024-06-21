This folder contains the code used in Abreu et al (Nature Ecology & Evolution, 2024).

The subfolder infer_fitness contains code for inferring fitness of mutants based on raw frequency data, and formatting these fitness inferences for plotting in the figures.

The files included here are:

# figure2.py:
This figure plots: A) non-additivity in three fluctuating environments, B-D) a few individual barcoded mutant trajectories (averaged across replicate fitness measurements) against the expected trajectories based on static environment fitness, E) all values of non-additivity across all mutants in all fluctuating environments, F) all values of environmental memory, G-H) non-additivity and memory significance scores.

# figure3.py:
This figure plots fitness in three fluctuating environments and their corresponding static environment components. The first row shows fitness across the pairs of static environments, where mutants are colored more red or more blue if they have higher fitness in either of the two environments. In the second row, this color scheme is preserved, but the plots show how the mutants' fitness changes from the static environment to the corresponding component of the fluctuating environment. For example, the left plot in the second row shows fitness in the galactose component in the fluctuating environment minus fitness in static galactose. These plots show that darker blue/red points tend to be farther from the origin, meaning mutants that have bigger differences in fitness across the static environments have more memory. The third row shows that this trend is even more noticeable when the mutants are colored not just by their difference in fitness across one pair of environments, but by their mean fitness difference across all ten pairs.

# figure4.py:
This figure

# figure5.py:

# angle_origin_distance.py:
