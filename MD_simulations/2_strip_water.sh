#!/bin/bash

#SBATCH --job-name=strip_water
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


# Activate the conda environment with MMGBSA and GROMACS tools
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate mmgbsa_ml

# Define the range of subdirectories to process under each MycMax* directory
start=1
end=20

# Loop over each subdirectory index for all MycMax* folders
for ((i=$start; i<=$end; i++)); do
    for dir in MycMax*/$i; do
        
        # Enter the subdirectory
        cd "$dir" || { echo "Cannot enter $dir"; exit 1; }

        # 1) Create an index group for protein + DNA (groups 1 and 4)
        echo -e "1 | 4 \n q \n" | gmx make_ndx -f npt.gro -o index_nowat.ndx

        # 2) Generate a trajectory without water, fixing periodic boundary conditions
        echo 22 | gmx trjconv -f npt.xtc -s npt.tpr -o npt_whole_nowat.xtc -n index_nowat.ndx -pbc whole &
        wait $!  # Wait for trjconv to finish

        # 3) Strip water and ions (SOL, NA, CL) from the topology file
        awk '
            /^\[ molecules \]/ {found_molecules=1}
            found_molecules && ($1 == "SOL" || $1 == "NA" || $1 == "CL") { next }
            { print }
        ' topol.top > topol_nowat.top

        # 4) Recreate a GRO file without water:
        #   a) Find the last atom index before any SOL entries in npt0.gro
        last_atom_number=$(
            awk '
                /^SOL/ && sol_found == 0 { sol_found=1; next }
                /^[0-9]+[A-Za-z][A-Za-z0-9 ]{4}/ && sol_found == 0 { last=$3 }
                END { print last }
            ' npt0.gro
        )
        last_atom_number=$(( last_atom_number - 1 ))

        #   b) Update the atom count on line 2, write to temp.gro
        awk -v last="$last_atom_number" 'NR==2 { $1=last } { print }' npt0.gro > temp.gro

        #   c) Retain all lines before the first SOL, and the final two lines (box vectors)
        awk '
            /SOL/ && !sol_line { sol_line=NR }
            { lines[NR]=$0 }
            END {
                for (j=1; j<sol_line; j++) print lines[j]
                print lines[NR-1]
                print lines[NR]
            }
        ' temp.gro > npt0_nowat.gro
        rm temp.gro

        # 5) Build a new TPR file without waters
        gmx grompp -f npt.mdp \
                   -c npt0_nowat.gro \
                   -r npt0_nowat.gro \
                   -p topol_nowat.top \
                   -o npt_nowat.tpr \
                   -maxwarn 1

        # Return to the parent directory
        cd - > /dev/null || exit
    done
done