# ML-Protein-DNA-Binding-Affinity

## Overview
This repository contains the scripts, input files, and documentation necessary for data processing, mutation generation, and energy calculations using molecular dynamics and MMGBSA methods.

## Directory Structure

```
ML-Protein-DNA-Binding-Affinity/
├── scripts/
│   ├─ process_gcPBM.ipynb
│   ├─ Mutate_PDB_w3dna.ipynb
│   └─ AMBER_MMGBSA.ipynb
├── inputs/
│   ├─ amber14sb_OL15.ff
│   ├─ initial_setup.sh
│   ├─ submit_equil.sh
│   ├─ submit_nvt.sh
│   ├─ submit_npt0.sh
│   ├─ submit_npt.sh
│   └─ submit_mmgbsa.sh
├── DNA_library/
├── run_MD.sh
└── run_mmgbsa.sh
```

---

## Directory Details

### scripts
This directory contains Jupyter Notebooks for processing data and extracting energy terms:

- **process_gcPBM.ipynb**:
  - **Description**: Processes experimental data from gcPBM.

- **Mutate_PDB_w3dna.ipynb**:
  - **Description**: Mutates DNA sequences using [web3DNA](http://web.x3dna.org/index.php/protein). The generated PDB files are outputted to the `../DNA_library` directory.

- **AMBER_MMGBSA.ipynb**:
  - **Description**: Extracts energy terms from MMGBSA output files using `gmx_MMPBSA` and a custom `vdw_mapping.csv` file containing van der Waals radii for each atom. Outputs `energy_corrections_df.csv`.

### inputs
This directory contains input files required for setting up and running simulations:

- **amber14sb_OL15.ff**:
  - **Description**: AMBER force field with OL15 topology.

- **initial_setup.sh**:
  - **Description**: Sets up the system, processes the PDB by adding topology parameters, creating a simulation box, and performing solvation and ionization. Requires `ions.mdp` (GROMACS parameter file for ionization).

- **submit_equil.sh**:
  - **Description**: Submission script for initial energy minimization using `minim.mdp` (GROMACS parameter file for steepest descent minimization).

- **submit_nvt.sh**:
  - **Description**: Submission script for NVT equilibration. Requires `nvt.mdp`.

- **submit_npt0.sh**:
  - **Description**: Submission script for restrained NPT equilibration. Requires `npt0.mdp`.

- **submit_npt.sh**:
  - **Description**: Submission script for unrestrained NPT run. Requires `npt.mdp`.

- **submit_mmgbsa.sh**:
  - **Description**: Submission script for running MMGBSA analysis. Requires `mmgbsa.in` (AMBER input file).

### DNA_library
This directory contains PDB files generated from the `Mutate_PDB_w3dna.ipynb` script for use in downstream analysis.

### run_MD.sh
- **Description**: A script that creates directories, copies necessary input files to the directories, and submits the job for molecular dynamics setup and execution.

### run_mmgbsa.sh
- **Description**: A script that navigates through each created directory and submits the job for MMGBSA analysis.

---
