#!/bin/bash
#
#SBATCH --job-name=equil
#SBATCH --output=out_equil.txt 
#SBATCH --account=jyu20_lab_gpu
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

module load gromacs/2022.1/gcc.8.4.0-cuda.11.7.1
export GMXLIB=/dfs6/pub/calmasri/ForceFields
sbatch --dependency=afterok:${SLURM_JOBID} submit_nvt.sh


gmx grompp -f minim.mdp -c protein_solv_ions.gro -p topol.top -o em.tpr -r protein_solv_ions.gro
gmx mdrun -ntmpi 1 -ntomp 10 -v -deffnm em