#!/bin/bash
#
#SBATCH --job-name=npt0
#SBATCH --output=out_npt0_%j.txt 
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

prev_out_file="out_npt0_${SLURM_JOBID_PREV}.txt"
resubmit_script="submit_npt0.sh"
next_script="submit_npt.sh"

if [ -z $SLURM_JOBID_PREV ]; then #first iteration should give empty variable, submit depenency
	echo "First iteration, submit dependency"
	sbatch --dependency=afternotok:${SLURM_JOBID} --export=SLURM_JOBID_PREV=${SLURM_JOBID} ${resubmit_script}
	
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

# start executing the program,
if [ -a npt0.cpt ]; then
	gmx mdrun -ntmpi 1 -ntomp 10 -deffnm npt0 -cpi npt0.cpt -nb gpu -pme gpu -bonded cpu
else
	gmx grompp -f npt0.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt0.tpr
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm npt0 -nb gpu -pme gpu -bonded cpu
fi
