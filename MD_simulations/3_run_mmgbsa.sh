#!/bin/bash

#SBATCH --job-name=mmpbsa
#SBATCH --output=out.txt
#SBATCH --partition=standard
#SBATCH --account=jyu20_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=calmasri@uci.edu

# Activate the conda environment with MMGBSA and GROMACS tools
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate mmgbsa_ml

# Define the range of subdirectories under each MycMax* folder
start=1
end=20

# ----------------------------------------------------------
# 1) Copy necessary MMGBSA input files into each simulation folder
# ----------------------------------------------------------
for ((i=$start; i<=$end; i++)); do
    for dir in MycMax*/$i; do
        # Copy the MMGBSA control input file
        cp inputs/mmgbsa_strain.in "$dir"
        # Copy the SLURM submission script for the MMGBSA run
        cp inputs/submit_mmgbsa_strain.sh "$dir"
        # Copy the force-field parameter files (Amber14SB + OL15)
        cp -r inputs/amber14sb_OL15.ff "$dir"
    done
done

# ----------------------------------------------------------
# 2) Submit the MMGBSA jobs in each directory
# ----------------------------------------------------------
for ((i=$start; i<=$end; i++)); do
    for dir in MycMax*/$i; do
        # Change into the target directory
        cd "$dir" || { echo "Cannot enter $dir"; exit 1; }

        # Submit the MMGBSA job via the provided SLURM script
        sbatch submit_mmgbsa_strain.sh

        # Return to the parent directory
        cd - > /dev/null || exit
    done
done
