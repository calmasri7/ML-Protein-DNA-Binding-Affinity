# Define the range of numbered subdirectories to create under each base directory
start=1
end=20

# Directory containing the source PDB files 
pdb_dir="../DNA_library_new"

# Directory whose contents (e.g., scripts, mdp files) should be copied into each subdirectory
inputs_dir="inputs"

# Loop over every .pdb file in pdb_dir
for pdb_file in "$pdb_dir"/*.pdb; do
    # Extract filename without extension, e.g. "MycMax_CACGTG"
    base_name=$(basename "$pdb_file" .pdb)
    echo "Processing .pdb file: $pdb_file (Base name: $base_name)"

    # Create a directory named after the base_name if it doesn't already exist
    if [ ! -d "$base_name" ]; then
        mkdir "$base_name"
        echo "Created directory: $base_name"
    fi

    # Within the base_name directory, create subdirectories numbered start…end
    for i in $(seq $start $end); do
        subdir="$base_name/$i"
        if [ ! -d "$subdir" ]; then
            mkdir "$subdir"
            echo "Created subdirectory: $subdir"
        fi

        # Copy the original PDB file into the new subdirectory
        cp "$pdb_file" "$subdir/"
        echo "Copied $pdb_file into $subdir/"

        # Copy all files from inputs_dir into the subdirectory
        cp -r "$inputs_dir"/* "$subdir/"
        echo "Copied contents of $inputs_dir into $subdir/"

        # Change directory into the newly populated subdirectory
        cd "$subdir" || { echo "Cannot change to directory $subdir"; exit 1; }

        # Submit the initial setup job if the script exists
        if [ -f "initial_setup.sh" ]; then
            echo "Submitting job in $subdir"
            sbatch initial_setup.sh
        else
            echo "initial_setup.sh not found in $subdir"
        fi

        # Return to the parent directory before the next iteration
        cd - > /dev/null || exit
    done
done