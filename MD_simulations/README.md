# MD Simulations

This folder contains everything needed to generate and analyse the 20‑replicate molecular‑dynamics (MD) simulations that underpin the MMGBSA + ML study of Myc/Max–DNA binding.  Each script is fully automated to run on a SLURM cluster and hands off to the next stage when successful, while robustly resubmitting itself after node pre‑emption or time‑limit failure.

---

## Directory Layout

```text
MD_simulations/
├── inputs/                 # All reusable input files
│   ├── amber14sb_OL15.ff/  # Amber14SB protein + OL15 DNA force‑field parameters
│   ├── initial_setup.sh    # Topology generation, boxing, solvation, ions
│   ├── ions.mdp            # grompp parameters for ion placement
│   ├── minim.mdp           # Steepest‑descent minimisation parameters
│   ├── nvt.mdp             # 2 ns 300 K NVT equilibration
│   ├── npt0.mdp            # 4 ns restrained NPT (heavy‑atom restraints)
│   ├── npt.mdp             # 8 ns unrestrained production NPT
│   ├── submit_equil.sh     # SLURM wrapper that runs `initial_setup.sh` + minim
│   ├── submit_nvt.sh       # SLURM wrapper for NVT step
│   ├── submit_npt0.sh      # SLURM wrapper for restrained NPT0 step
│   ├── submit_npt.sh       # SLURM wrapper for production NPT step
│   ├── mmgbsa_strain.in    # gm​x_MMPBSA control file (strain‑energy variant)
│   ├── submit_mmgbsa_strain.sh
│   └── submit_naccess.sh   # SLURM wrapper for NACCESS SASA analysis
├── 1_run_MD.sh             # Creates 20 replicated run directories & launches MD
├── 2_strip_water.sh        # Generates no‑water topologies / trajectories for MMGBSA
├── 3_run_mmgbsa.sh         # Copies MMGBSA inputs and submits jobs
├── 4_run_naccess.sh        # Creates PDB snapshots & submits SASA jobs
└── MycMax_<SEQ>*/          # Auto‑generated: one folder per DNA sequence
    └── 1 … 20/             # Twenty replicate runs per sequence
```

*Sample data*: `MycMax_TTTTTTTTTTTTTTGAGAAAATGAAGACAATTATCT/` shows the full directory tree after an MD/analysis run (select outputs included).

---

## Script‑by‑Script Details

### 1\_run\_MD.sh  — **setup & launch MD**

|                              | Description                                                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**                  | Create replicate sub‑folders, copy inputs, and submit the first SLURM job (`initial_setup.sh`).                                       |
| **Key inputs**               | • `../DNA_library_new/*.pdb` reference complexes                                                                                      |
| 							   | • everything under `inputs/`                                                                                                          |
| **Key outputs**              | Each `MycMax_<SEQ>/<rep>/` now contains the PDB, all `.mdp` & submission scripts, and a submitted SLURM job for topology + solvation. |

### initial\_setup.sh  (run automatically)

Generates the topology and solvated system.

| Step           | Output file                                                | Notes                                                    |
| -------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| Topology build | `prot_processed.pdb`, `protein_processed.gro`, `topol.top` | Removes terminal phosphates to avoid missing parameters. |
| Boxing         | `protein_newbox.gro`                                       | 1.2 nm rectangular box.                                  |
| Solvation      | `protein_solv.gro`                                         | TIP3P water.                                             |
| Ions           | `ions.tpr`, `protein_solv_ions.gro`                        | Neutralised, 0.15 M NaCl.                                |
| Minimisation   | `em.tpr`, `em.log`, `em.gro`, …                            | Steepest‑descent to 1000 kJ mol⁻¹ nm⁻¹.                  |

SLURM standard‑out is captured in `out_equil_${SLURM_JOBID}.txt`.

### submit\_nvt.sh → **NVT 300 K equilibration**

Continues automatically after `submit_equil.sh` finishes OK, or resubmits itself if the previous job was pre‑empted.  Produces `nvt.tpr`, `nvt.cpt`, `nvt.trr`, `nvt.edr`, `nvt.gro`, and SLURM log `out_nvt_<JOBID>.txt`.

### submit\_npt0.sh → **restrained NPT (4 ns)**

Heavy‑atom position restraints (1000 kJ mol⁻¹ nm⁻²).  Checkpoints (`npt0.cpt`) allow seamless restarts.  Output analogous to NVT step (`npt0.*`).

### submit\_npt.sh → **production NPT (8 ns)**

Unrestrained MD.  Auto‑resubmits on pre‑emption or wall‑time expiry.  Final coordinate / trajectory files analogous to NVT step (`npt.*`).

### 2\_strip\_water.sh  — **prepare no‑water inputs**

| Output                | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `index_nowat.ndx`     | Group **1 Protein** + **4 DNA** for MMGBSA and SASA. |
| `npt_whole_nowat.xtc` | Whole PBC‑fixed trajectory, water/ions removed.      |
| `topol_nowat.top`     | Topology stripped of `SOL`, `NA`, `CL`.              |
| `npt0_nowat.gro`      | Matching solvent‑free coordinate file.               |
| `npt_nowat.tpr`       | Re‑generated run‑input for MMGBSA (no water).        |

### 3\_run\_mmgbsa.sh  — **launch MMGBSA**

Copies `mmgbsa_strain.in`, forcefield directory, and SLURM wrapper into each replicate folder, then submits `submit_mmgbsa_strain.sh`, which executes the MMGBSA runs


### 4\_run\_naccess.sh  — **SASA analysis**

Generates frame‑by‑frame PDBs for `DNA`, `Protein`, and `DNA_Protein` groups, copies `submit_naccess.sh`, and submits an NACCESS job that outputs standard `.rsa` and `.asa` files alongside a `.log` summary per frame.

---

## Workflow Overview

1. **Run setup** `1_run_MD.sh` – creates replicate directories and submits `initial_setup.sh`.
2. **Equilibration chain** – `submit_equil.sh` → `submit_nvt.sh` → `submit_npt0.sh` (auto‑chained).
3. **Production MD** – `submit_npt.sh` (auto‑submitted after NPT0).
4. **Post‑processing** – `2_strip_water.sh` to obtain solvent‑free data.
5. **Energy analysis** – `3_run_mmgbsa.sh` launches MMGBSA per replicate.
6. **Surface analysis** – `4_run_naccess.sh` launches NACCESS per replicate.

All scripts record SLURM output in `out_<stage>_<JOBID>.txt` and resubmit themselves automatically if pre‑empted.

---

For questions, contact **Carmen Al Masri** ([calmasri@uci.edu](mailto:calmasri@uci.edu)).
