#!/usr/bin/env python
"""
hyperopt.py 

Bayesian hyper-parameter optimization of a PyTorch neural-network workflow
(using **Hyperopt** with the **Tree-Parzen Estimator** algorithm and distributed
with **SparkTrials**).

Workflow per Hyperopt trial
----------------------------------
1. **Data prep** – calls an external script (`torch_prep_kfold.py`) in  
   `"--process train"` mode to build repeated K-fold train/validation CSVs
   (scramble fraction fixed at **0.0** during tuning).
2. **Training** – calls another script (`run_model.py`) in training mode  
   (`--mode 0`), which itself performs the repeated K-fold training/evaluation
   and writes a summary metrics CSV (`final_metrics_*_trn_scr0p00.csv`).
3. **Objective value** – reads that metrics CSV and converts it to a single
   scalar loss (lower is better) that Hyperopt minimises.
4. **SparkTrials** – runs several trials *in parallel* on a Spark cluster /
   local Spark session (parallelism controlled by `--spark_parallelism`).

"""

###############################################################################
# Imports
###############################################################################
import os, csv, json, logging, argparse, subprocess
from typing import Any

import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, SparkTrials      # Bayesian optimisation
from pyspark.sql import SparkSession                # parallel backend

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# Argument parsing
###############################################################################
def parse_args():
    """
    Build the CLI.  
    """
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian hyper-parameter optimisation for a neural-network model.\n"
            "Assumes train/test split is done externally; this wrapper first "
            "runs torch_prep_kfold.py (data prep) then run_model.py (training)."
        )
    )

    # ---------- external script paths ---------------------------------- #
    parser.add_argument("--prep_script",  default="torch_prep_kfold.py",
                        help="Data-prep script path.")
    parser.add_argument("--train_script", default="run_model.py",
                        help="Training script path (performs repeated K-fold).")

    # ---------- I/O directories / files -------------------------------- #
    parser.add_argument("--data_dir",  default="Data",
                        help="Dir for CSV folds (created / reused).")
    parser.add_argument("--model_dir", default="Model",
                        help="Dir to store nn checkpoints.")
    parser.add_argument("--reference_file", default="../Inputs/exp_data_all.csv",
                        help="Experimental reference CSV (if required by prep).")

    # ---------- column names ------------------------------------------ #
    parser.add_argument("--ref_id_col",    default="sequence")
    parser.add_argument("--ref_label_col", default="label")
    parser.add_argument("--feature_id_col", default="sequence")

    # ---------- feature CSV(s) ---------------------------------------- #
    parser.add_argument("--filenames", nargs="+", required=True,
                        help="One or more feature CSV paths.")
    parser.add_argument("--usecols", nargs="+",
                        help="Optional subset of cols to read (must include ID).")

    # ---------- general prep params ----------------------------------- #
    parser.add_argument("--keep_last_percent", type=float, default=0.0,
                        help="Keep last X%% of rows per group (initial guess).")
    parser.add_argument("--random_state", type=int, default=42)

    # ---------- model/task specifics ---------------------------------- #
    parser.add_argument("--model_type", choices=["reg", "bin", "mclass"],
                        required=True)
    parser.add_argument("--data_scale", choices=["log", "nonlog"], default="log")
    parser.add_argument("--use_early_stopping", action="store_true")

    # ---------- Spark / Hyperopt -------------------------------------- #
    parser.add_argument("--spark_parallelism", type=int, default=4,
                        help="#parallel workers for SparkTrials.")
    parser.add_argument("--max_evals", type=int, default=50,
                        help="#total Hyperopt trials.")

    # ---------- repeated K-fold params (forwarded) -------------------- #
    parser.add_argument("--kfold",       type=int, default=5)
    parser.add_argument("--num_repeats", type=int, default=5)

    # ---------- baseline NN hyper-params (tuned by Hyperopt) ---------- #
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--hidden_size",   type=int, default=44)
    parser.add_argument("--lrn_rate",      type=float, default=1e-4)
    parser.add_argument("--wt_decay",      type=float, default=1e-4)
    parser.add_argument("--dropout_input_output", type=float, default=0.1)
    parser.add_argument("--dropout_hidden",       type=float, default=0.1)
    parser.add_argument("--navg", type=int, default=80,
                        help="#rows to average per sequence (initial guess).")

    # ---------- misc --------------------------------------------------- #
    parser.add_argument("--prefix",      default="gbsa",
                        help="Prefix used by data-prep & training scripts.")
    parser.add_argument("--output_file", default="predictions",
                        help="Prefix for per-fold prediction files.")

    return parser.parse_args()

###############################################################################
# Fixed scramble fraction for tuning
###############################################################################
SCR_FRACTION = 0.0
FRAC_STR = f"{SCR_FRACTION:.2f}".replace(".", "p")   # "0p00"

###############################################################################
# Hyperopt search space – uniform / log-uniform priors
###############################################################################
search_space = {
    # data-prep
    'keep_last_percent':    hp.quniform('keep_last_percent', 5, 100, 5),
    'navg':                 hp.quniform('navg', 1, 400, 20),

    # model architecture
    'hidden_layers':        hp.choice('hidden_layers', [1, 2, 3, 4, 5, 6]),
    'hidden_size':          hp.quniform('hidden_size', 4, 116, 4),

    # optimiser / regularisation
    'lrn_rate':             hp.loguniform('lrn_rate', np.log(1e-5), np.log(1e-2)),
    'wt_decay':             hp.loguniform('wt_decay', np.log(1e-10), np.log(1e-3)),

    # dropout
    'dropout_input_output': hp.uniform('dropout_input_output', 0.0, 0.6),
    'dropout_hidden':       hp.uniform('dropout_hidden',       0.0, 0.6),
}

###############################################################################
# ---------- helper: call data-prep script ---------------------------------- #
###############################################################################
def run_data_prep(keep_last_percent: float, navg: int, args):
    """
    Execute *torch_prep_kfold.py* to generate train/val CSVs for the given
    hyper-parameters.  All stdout/stderr is captured; non-zero exit aborts trial.
    """
    original_dir = os.getcwd()
    os.chdir(args.data_dir)                      # prep script expects CWD=Data
    try:
        cmd = [
            "python", args.prep_script,
            "--process", "train",
            "--reference_file",   args.reference_file,
            "--ref_id_col",       args.ref_id_col,
            "--ref_label_col",    args.ref_label_col,
            "--feature_id_col",   args.feature_id_col,
            "--keep_last_percent", str(keep_last_percent),
            "--model_type",       args.model_type,
            "--navg",             str(int(navg)),
            "--random_state",     str(args.random_state),
            "--prefix",           args.prefix,
            "--scramble_fractions", str(SCR_FRACTION),
            "--kfold",            str(args.kfold),
            "--num_repeats",      str(args.num_repeats)
        ]
        if args.filenames:
            cmd += ["--filenames"] + args.filenames
        if args.usecols:
            cmd += ["--usecols"] + args.usecols

        logging.info(f"[DataPrep] {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
        if res.returncode != 0:
            logging.error(f"Data-prep failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
            raise RuntimeError("Data-prep script exited with error.")
        logging.debug(f"Data-prep STDOUT:\n{res.stdout}")
    finally:
        os.chdir(original_dir)

###############################################################################
# ---------- helper: call training script ----------------------------------- #
###############################################################################
def run_training_evaluation(
    *,
    keep_last_percent: float,
    navg: int,
    hidden_size: int,
    hidden_layers: int,
    lrn_rate: float,
    wt_decay: float,
    dropout_input_output: float,
    dropout_hidden: float,
    args
) -> dict:
    """
    Execute *run_model.py* in training mode (`--mode 0`) and parse the resulting
    metrics CSV.  Returns a dict of key metrics for objective calculation.
    """
    cmd = [
        "python", args.train_script,
        "--mode", "0",
        "--model_type",              args.model_type,
        "--data_scale",              args.data_scale,

        "--lrn_rate",                str(lrn_rate),
        "--wt_decay",                str(wt_decay),
        "--dropout_input_output",    str(dropout_input_output),
        "--dropout_hidden",          str(dropout_hidden),
        "--hidden_size",             str(hidden_size),
        "--hidden_layers",           str(hidden_layers),

        "--prefix",                  args.prefix,
        "--data_dir",                args.data_dir,
        "--model_dir",               args.model_dir,
        "--output_file",             args.output_file,
        "--ref_id_col",              args.ref_id_col,
        "--ref_label_col",           args.ref_label_col,

        "--scramble_fractions",      str(SCR_FRACTION),
        "--kfold",                   str(args.kfold),
        "--num_repeats",             str(args.num_repeats)
    ]
    if args.use_early_stopping:
        cmd.append("--use_early_stopping")

    logging.info(f"[Training] {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        logging.error(f"Training failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        raise RuntimeError("Training script exited with error.")

    # ------------------------------------------------------------------ #
    # Parse summary metrics CSV written by run_model.py
    # ------------------------------------------------------------------ #
    metrics_csv = f"final_metrics_{args.model_type}_trn_scr{FRAC_STR}.csv"
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"{metrics_csv} not found after training.")

    with open(metrics_csv) as f:
        row = next(csv.DictReader(f))

    # convert row → metric dict depending on task
    if args.model_type == "reg":
        metrics = {
            "MSE":  float(row["MSE"]),
            "Pear": float(row["Pear"]) if row["Pear"] not in ["", "nan"] else np.nan
        }
    else:
        metrics = {
            "Accuracy": float(row["Accuracy"]),
            "MCC":      float(row["MCC"])
        }
    return metrics

###############################################################################
# ---------- objective passed to Hyperopt ----------------------------------- #
###############################################################################
def evaluate_hyperparams(params: dict, args) -> float:
    """
    Hyperopt objective wrapper:
      * Runs data-prep + training
      * Converts resulting metrics to a scalar loss (to minimise)
      * Returns large sentinel loss on failure / NaNs
    """
    # cast hyperopt floats (might come as np.float64) → python native
    keep_last = float(params['keep_last_percent'])
    navg      = int(params['navg'])
    layers    = int(params['hidden_layers'])
    hsize     = int(params['hidden_size'])
    lr        = float(params['lrn_rate'])
    wd        = float(params['wt_decay'])
    d_io      = float(params['dropout_input_output'])
    d_hd      = float(params['dropout_hidden'])

    # 1. Data prep
    try:
        run_data_prep(keep_last, navg, args)
    except Exception as e:
        logging.error(f"[Objective] Data-prep failed (keep_last={keep_last}, navg={navg}): {e}")
        return 1e6   # large penalty

    # 2. Training / parse metrics
    try:
        metrics = run_training_evaluation(
            keep_last_percent = keep_last,
            navg              = navg,
            hidden_size       = hsize,
            hidden_layers     = layers,
            lrn_rate          = lr,
            wt_decay          = wd,
            dropout_input_output = d_io,
            dropout_hidden    = d_hd,
            args = args
        )
    except Exception as e:
        logging.error(f"[Objective] Training failed: {e}")
        return 1e6

    # 3. Convert to scalar loss
    if args.model_type == "reg":
        mse, pear = metrics["MSE"], metrics["Pear"]
        if np.isnan(mse) or np.isnan(pear):
            return 1e6
        loss = 0.5 * mse + 0.5 * (1.0 - pear)
    else:
        acc, mcc = metrics["Accuracy"], metrics["MCC"]
        if np.isnan(acc) or np.isnan(mcc):
            return 1e6
        loss = 0.5 * (1.0 - acc) + 0.5 * (1.0 - mcc)

    logging.info(
        f"[Objective] params={params}, metrics={metrics}, loss={loss:.4f}"
    )
    return loss

###############################################################################
# Main entry-point
###############################################################################
def main():
    args = parse_args()

    # ------------------- Spark session --------------------------------- #
    spark = (SparkSession.builder
             .appName("HyperoptNN")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())

    trials = SparkTrials(parallelism=args.spark_parallelism)
    hidden_choices = [1, 2, 3, 4, 5, 6]   # must match search_space choice order

    # ------------------- run Hyperopt ---------------------------------- #
    best_raw = fmin(
        fn=lambda p: evaluate_hyperparams(p, args),
        space=search_space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        verbose=False
    )
    logging.info(f"Best Hyperopt result indices ≈ {best_raw}")

    # translate choice index → actual hidden_layers
    best_cfg = {
        "keep_last_percent":    float(best_raw['keep_last_percent']),
        "navg":                 int(best_raw['navg']),
        "hidden_layers":        hidden_choices[ best_raw['hidden_layers'] ],
        "hidden_size":          int(best_raw['hidden_size']),
        "lrn_rate":             float(best_raw['lrn_rate']),
        "wt_decay":             float(best_raw['wt_decay']),
        "dropout_input_output": float(best_raw['dropout_input_output']),
        "dropout_hidden":       float(best_raw['dropout_hidden']),
        "model_type":           args.model_type,
        "data_scale":           args.data_scale,
        "kfold":                args.kfold,
        "num_repeats":          args.num_repeats
    }

    # persist best config
    with open(f"best_hyperparams_{args.model_type}.json", "w") as f:
        json.dump(best_cfg, f, indent=4)
    logging.info("Best hyper-parameters saved.")

    # ------------------- collect all trials ---------------------------- #
    trial_rows = []
    for t in trials.trials:
        res  = t['result']
        vals = t['misc']['vals']
        hl_idx = vals.get('hidden_layers', [None])[0]

        trial_rows.append({
            "loss":   res.get('loss'),
            "status": res.get('status'),
            "keep_last_percent": vals['keep_last_percent'][0],
            "navg":              vals['navg'][0],
            "hidden_layers":     hidden_choices[hl_idx] if hl_idx is not None else None,
            "hidden_size":       vals['hidden_size'][0],
            "lrn_rate":          vals['lrn_rate'][0],
            "wt_decay":          vals['wt_decay'][0],
            "dropout_input_output": vals['dropout_input_output'][0],
            "dropout_hidden":       vals['dropout_hidden'][0]
        })

    pd.DataFrame(trial_rows).to_csv(f"hyperopt_results_{args.model_type}.csv", index=False)
    logging.info("All trial results saved.")

    spark.stop()


if __name__ == "__main__":
    main()
