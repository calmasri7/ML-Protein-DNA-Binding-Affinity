#!/bin/bash
#
#SBATCH --job-name=nvt
#SBATCH --output=out_nvt_%j.txt  # Using %j to include job ID in output filename
#SBATCH --account=jyu20_lab_gpu
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

export GMXLIB=/dfs6/pub/calmasri/ForceFields
module load gromacs/2022.1/gcc.8.4.0-cuda.11.7.1

# Accept SLURM_JOBID_PREV as an argument
echo "Previous slurm jobID: ${SLURM_JOBID_PREV}"

prev_out_file="out_nvt_${SLURM_JOBID_PREV}.txt"
resubmit_script="submit_nvt.sh"
next_script="submit_npt0.sh"

if [ -z $SLURM_JOBID_PREV ]; then #first iteration should give empty variable, submit depenency
	echo "First iteration, submit dependency"
	sbatch --dependency=afternotok:${SLURM_JOBID} --export=SLURM_JOBID_PREV=${SLURM_JOBID} ${resubmit_script} #current job ID passed down to next submission
	
else 
	if [ -f "${prev_out_file}" ] && grep -q "PREEMPTION" "${prev_out_file}"; then #Starting from the second iteration, only submit dependency if error due to preemption
    	echo "Preemption detected, resubmitting"
    	sbatch --dependency=afternotok:${SLURM_JOBID} --export=SLURM_JOBID_PREV=${SLURM_JOBID} ${resubmit_script} #current job ID passed down to next submission
	else
		"Error is not due to preemption, exit"
		exit 1
    fi
    
fi
sbatch --dependency=afterok:${SLURM_JOBID} ${next_script} #submit either way, if the job fails dependency won't be satisfied

# Start executing the program
if [ -a nvt.cpt ]; then
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm nvt -cpi nvt.cpt -nb gpu -pme gpu -bonded cpu
else
    gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm nvt -nb gpu -pme gpu -bonded cpu
fi