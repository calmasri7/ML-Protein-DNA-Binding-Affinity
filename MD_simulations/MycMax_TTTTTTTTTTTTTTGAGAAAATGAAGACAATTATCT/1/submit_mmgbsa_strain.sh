#!/bin/bash

#SBATCH --job-name=mmpbsa
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


eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate gmxMMPBSA

# Run MMGBSA
mpirun -np 10 gmx_MMPBSA MPI -O -i mmgbsa_strain.in -cs npt_nowat.tpr -cp topol_nowat.top -ci index_nowat.ndx -cg 1 4 -ct npt_whole_nowat.xtc -o FINAL_RESULTS_strain.dat -eo FINAL_RESULTS_strain.csv -do FINAL_RESULTS_DECOMP_strain.dat -deo FINAL_RESULTS_DECOMP_strain.csv &> out.txt &
wait $!