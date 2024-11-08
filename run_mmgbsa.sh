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

# Activate the conda environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate gmxMMPBSA

# Define the range of directories to process
start=1
end=10

# Loop to copy input files and submit jobs
for ((i=1 ; i<=10; i++)); do
    for dir in MycMax*/$i; do
        # Copy input files into each directory
        cp inputs/mmgbsa_strain.in inputs/mmgbsa_nostrain.in $dir
        cp inputs/submit_mmgbsa_strain.sh inputs/submit_mmgbsa_nostrain.sh $dir
        cp -r inputs/amber14sb_OL15.ff $dir

        # Move into the directory
        cd $dir || exit

        # Run GROMACS commands
        echo -e "q \n" | gmx make_ndx -f npt.gro
        echo 0 | gmx trjconv -f npt.xtc -s npt.tpr -o npt_whole.xtc -pbc whole &

        # Wait for background process to finish
        wait $!

        # Submit MMGBSA jobs
        sbatch submit_mmgbsa_strain.sh
        sbatch submit_mmgbsa_nostrain.sh

        # Move back to the original directory
        cd - || exit
    done
done

# Loop through directories in the current location and submit jobs within range
for dir in */*/; do
    # Extract the directory number from the path
    dir_number=$(basename "$dir")

    # Check if the directory is within the range
    if [[ "$dir_number" -ge "$start" && "$dir_number" -le "$end" ]]; then
        echo "Processing directory: $dir"

        # Move into the directory
        cd "$dir" || exit

        # Submit the non-strain MMGBSA job
        sbatch submit_mmgbsa_nostrain.sh

        # Go back to the original directory
        cd - || exit
    fi
done