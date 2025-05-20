#!/bin/bash

#SBATCH --job-name=naccess
#SBATCH --output=out.txt
#SBATCH --partition=standard
#SBATCH --account=jyu20_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=calmasri@uci.edu

# Initialize the conda environment for MMGBSA/analysis
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate mmgbsa_ml

# Define the parent directory containing MD simulation subfolders
parent_dir='parent_dir/MD_simulations'

# Loop over the simulation replicates
for ((i=1; i<=20; i++)); do
    # For each MycMax* folder and replicate number
    for dir in "$parent_dir"/MycMax*/$i; do
        
        # Only proceed if the directory exists
        if [ -d "$dir" ]; then
            # Change into the target directory
            pushd "$dir" > /dev/null
            
            # Create (if needed) and enter an 'naccess' subdirectory
            mkdir -p naccess
            cd naccess
            
            # ----------------------------------------------------------
            # 1) Split trajectory into PDB frames for DNA, protein, and complex
            # ----------------------------------------------------------
            # - “DNA” group → dna.pdb
            # - “Protein” group → protein.pdb
            # - “DNA_Protein” group → complex.pdb
            # All processes run in background, then 'wait' synchronizes them.
            echo DNA         | gmx trjconv -f ../npt_whole_nowat.xtc -s ../npt_nowat.tpr -o dna.pdb     -sep -n ../index_nowat.ndx &
            echo Protein     | gmx trjconv -f ../npt_whole_nowat.xtc -s ../npt_nowat.tpr -o protein.pdb -sep -n ../index_nowat.ndx &
            echo DNA_Protein | gmx trjconv -f ../npt_whole_nowat.xtc -s ../npt_nowat.tpr -o complex.pdb -sep -n ../index_nowat.ndx &
            
            # Wait for all three trjconv commands to finish
            wait
            
            # ----------------------------------------------------------
            # 2) Copy and submit the NACCESS analysis job
            # ----------------------------------------------------------
            # Copy the SLURM submission script from the parent inputs folder
            cp "$parent_dir/inputs/submit_naccess.sh" submit_naccess.sh
            # Submit the job to compute solvent-accessible surface areas
            sbatch submit_naccess.sh
            
            # Return to the previous directory
            popd > /dev/null
        else
            echo "Directory $dir does not exist, skipping."
        fi
    done
done
		
		
		
		
