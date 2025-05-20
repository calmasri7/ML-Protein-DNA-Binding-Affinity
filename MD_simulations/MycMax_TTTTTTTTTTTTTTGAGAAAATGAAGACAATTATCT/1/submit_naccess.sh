#!/bin/bash
#SBATCH --job-name=naccess
#SBATCH --account=jyu20_lab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=calmasri@uci.edu

# Path to the naccess executable
naccess='/dfs6/pub/calmasri/naccess/Naccess/naccess'

# Loop over PDB files and run naccess
for pdb_file in complex*.pdb prot*.pdb dna*.pdb; 
do
    echo "Running naccess on $pdb_file"
    $naccess $pdb_file
done

echo "All PDB files processed."