#!/bin/bash
#
#SBATCH --job-name=npt
#SBATCH --output=out_npt_%j.txt 
#SBATCH --account=jyu20_lab_gpu
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1

export GMXLIB=/dfs6/pub/calmasri/ForceFields
module load gromacs/2022.1/gcc.8.4.0-cuda.11.7.1

# Accept SLURM_JOBID_PREV as an argument
echo "Previous slurm jobID: ${SLURM_JOBID_PREV}"

prev_out_file="out_npt_${SLURM_JOBID_PREV}.txt"
resubmit_script="submit_npt.sh"

if [ -z $SLURM_JOBID_PREV ]; then # First iteration should give empty variable, submit dependency
    echo "First iteration, submit dependency"
    sbatch --dependency=afternotok:${SLURM_JOBID} --export=SLURM_JOBID_PREV=${SLURM_JOBID} ${resubmit_script}
else 
    if [ -f "${prev_out_file}" ] && (grep -q "PREEMPTION" "${prev_out_file}" || grep -q "TIME LIMIT" "${prev_out_file}"); then
        # Preemption or Time limit exceeded
        echo "Preemption or Time limit exceeded, resubmitting"
        sbatch --dependency=afternotok:${SLURM_JOBID} --export=SLURM_JOBID_PREV=${SLURM_JOBID} ${resubmit_script}
    else
        echo "Error is not due to preemption or time limit, exit"
        exit 1
    fi
fi

if [ -a npt.cpt ]; then # Run from checkpoint if it exists
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm npt -cpi npt.cpt -nb gpu -pme gpu -bonded cpu
else
    gmx grompp -f npt.mdp -c npt0.gro -r npt0.gro -t npt0.cpt -p topol.top -o npt.tpr
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm npt -nb gpu -pme gpu -bonded cpu
fi
