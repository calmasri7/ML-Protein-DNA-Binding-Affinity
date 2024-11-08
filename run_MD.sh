#!/bin/bash

# Define the range of subdirectories to create
start=1
end=10

# Path to the inputs directory and DNA library
inputs_dir="inputs"
dna_library_dir="DNA_library"

# Loop over all .pdb files in the DNA_library directory
for pdb_file in "$dna_library_dir"/*.pdb; do
    # Extract the base name of the .pdb file without the extension
    base_name=$(basename "$pdb_file" .pdb)
    
    # Create a main directory with the same name as the .pdb file
    main_dir="$base_name"
    mkdir -p "$main_dir"

    # Loop to create subdirectories, copy files, and submit jobs
    for ((i=$start; i<=$end; i++)); do
        sub_dir="$main_dir/$i"
        mkdir -p "$sub_dir"

        # Copy input files and the .pdb file into each subdirectory
        cp -r "$inputs_dir"/* "$sub_dir"
        cp "$pdb_file" "$sub_dir"

        # Change to the subdirectory
        cd "$sub_dir" || exit

        # Run sbatch on initial_setup.sh if it exists
        if [ -f "initial_setup.sh" ]; then
            echo "Running sbatch for $sub_dir"
            sbatch initial_setup.sh
        else
            echo "initial_setup.sh not found in $sub_dir"
        fi

        # Return to the original directory
        cd - || exit
    done
done