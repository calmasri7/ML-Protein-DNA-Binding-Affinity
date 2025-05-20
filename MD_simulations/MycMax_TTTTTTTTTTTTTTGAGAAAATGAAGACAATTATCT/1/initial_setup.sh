#!/bin/bash
#
#SBATCH --job-name=equil0
#SBATCH --output=equil0.txt 
#SBATCH --account=JYU20_LAB
#
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5000
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00


export GMXLIB=/dfs6/pub/calmasri/ForceFields
module load gromacs/2022.1/gcc.8.4.0-cuda.11.7.1
sbatch --dependency=afterok:${SLURM_JOBID} submit_equil.sh


pdb_file=$(find . -type f -name "MycMax*.pdb")
output_pdb="prot_processed.pdb"

# Need to remove the first 3 atoms from both chain A and chain B
awk '
    BEGIN { a_count = 0; b_count = 0; }

    {# Skip first 3 atoms of chain A
     if ($5 == "A" && a_count < 3 && ($3 == "P" || $3 == "O1P" || $3 == "O2P")) {
         a_count++;
         next;
     }
        
     # Skip first 3 atoms of chain B
     if ($5 == "B" && b_count < 3 && ($3 == "P" || $3 == "O1P" || $3 == "O2P")) {
          b_count++;
          next;
     }

     # Print the rest of the lines
     print;
    }
' "$pdb_file" > "$output_pdb"


echo 1 | gmx pdb2gmx -f $output_pdb -o protein_processed.gro -ss -merge all -water tip3p -renum yes -ignh &
wait $!


echo DNA | gmx editconf -f protein_processed.gro -o protein_newbox.gro -princ yes -c -d 1.2 
wait $!

gmx solvate -cp protein_newbox.gro -o protein_solv.gro -p topol.top
wait $!

gmx grompp -f ions.mdp -c protein_solv.gro -p topol.top -o ions.tpr -maxwarn 1
echo SOL | gmx genion -s ions.tpr -o protein_solv_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15 &
wait $!