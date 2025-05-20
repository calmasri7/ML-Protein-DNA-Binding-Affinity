#!/usr/bin/env python
"""
hyperopt.py  

Bayesian hyper-parameter optimisation wrapper for Support-Vector Machine
models.  Uses Hyperopt + the TPE algorithm and executes trials in parallel
via SparkTrials.

Per trial workflow
------------------
1. Data preparation  
   Calls an external script (`torch_prep_kfold.py`) with
   `--process train`, regenerating K-fold CSV splits (no label scrambling).
2. Training + cross-validation  
   Runs `run_model.py --mode 0`, which trains an SVM for each fold and writes a
   CSV summarising averaged metrics across folds.
3. Objective  
   Reads that CSV and converts metrics into a single scalar *loss*
   (lower is better) for Hyperopt.
4. SparkTrials  
   Executes many trials concurrently on a local/cluster Spark backend.

"""

###############################################################################
# Imports
###############################################################################
import os, csv, json, logging, argparse, subprocess
from typing import Any

import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, SparkTrials
from pyspark.sql import SparkSession

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# CLI
###############################################################################
def parse_args():
    """
    Build and parse command-line arguments. 
    """
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian hyper-parameter optimisation for an SVM model.\n"
            "Delegates data prep to torch_prep_kfold.py and training / CV to "
            "run_model.py.  Assumes an initial train/test split exists."
        )
    )

    # ---------- external script paths ---------------------------------- #
    parser.add_argument("--prep_script",  default="torch_prep_kfold.py")
    parser.add_argument("--train_script", default="run_model.py")

    # ---------- path / file parameters --------------------------------- #
    parser.add_argument("--data_dir",  default="Data")
    parser.add_argument("--model_dir", default="Model")
    parser.add_argument("--reference_file", default="../Inputs/exp_data_all.csv")

    # ---------- column names & feature CSVs ---------------------------- #
    parser.add_argument("--ref_id_col",    default="sequence")
    parser.add_argument("--ref_label_col", default="label")
    parser.add_argument("--filenames", nargs="+", required=True)
    parser.add_argument("--feature_id_col", default="sequence")
    parser.add_argument("--usecols", nargs="+")

    # ---------- data-prep tweaks -------------------------------------- #
    parser.add_argument("--keep_last_percent", type=float, default=0.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--navg", type=int, default=80)

    # ---------- task specifics ---------------------------------------- #
    parser.add_argument("--model_type", choices=["reg", "bin", "mclass"], required=True)
    parser.add_argument("--data_scale", choices=["log", "nonlog"], default="log")

    # ---------- misc flags ------------------------------------------- #
    parser.add_argument("--use_early_stopping", action="store_true")  # kept for API parity
    parser.add_argument("--spark_parallelism", type=int, default=4)
    parser.add_argument("--max_evals", type=int, default=50)

    # ---------- SVM baseline hyper-params (to be tuned) --------------- #
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--C",      type=float, default=1.0)
    parser.add_argument("--gamma",  type=float, default=0.1)

    # ---------- K-fold forwarding ------------------------------------ #
    parser.add_argument("--prefix",      default="gbsa")
    parser.add_argument("--output_file", default="predictions")
    return parser.parse_args()

###############################################################################
# Hyperopt search space (SVM-specific)
###############################################################################
search_space = {
    "keep_last_percent": hp.quniform("keep_last_percent", 5, 100, 5),
    "navg":              hp.quniform("navg", 1, 400, 20),

    # SVM kernel and numeric hyper-parameters
    "kernel": hp.choice("kernel", ["linear", "rbf", "poly", "sigmoid"]),
    "C":      hp.loguniform("C", np.log(1e-3), np.log(1e3)),
    "gamma":  hp.loguniform("gamma", np.log(1e-4), np.log(1.0))
}

###############################################################################
# Constant scramble fraction (no label scrambling during tuning)
###############################################################################
SCR_FRACTION = 0.0
FRAC_STR = f"{SCR_FRACTION:.2f}".replace(".", "p")     # "0p00"

###############################################################################
# ---------------------- helper: run data-prep ------------------------------ #
###############################################################################
def run_data_prep(keep_last_percent: float, navg: int, args):
    """
    Execute *torch_prep_kfold.py* within `args.data_dir`.  Any non-zero exit
    aborts the current Hyperopt trial.
    """
    orig_cwd = os.getcwd()
    os.chdir(args.data_dir)
    try:
        cmd = [
            "python", args.prep_script,
            "--process", "train",
            "--reference_file",    args.reference_file,
            "--ref_id_col",        args.ref_id_col,
            "--ref_label_col",     args.ref_label_col,
            "--feature_id_col",    args.feature_id_col,
            "--keep_last_percent", str(keep_last_percent),
            "--model_type",        args.model_type,
            "--navg",              str(int(navg)),
            "--random_state",      str(args.random_state),
            "--prefix",            args.prefix,
            "--scramble_fractions", str(SCR_FRACTION)
        ]
        if args.filenames: cmd += ["--filenames"] + args.filenames
        if args.usecols:   cmd += ["--usecols"]   + args.usecols

        logging.info(f"[DataPrep] {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
        if res.returncode != 0:
            logging.error(f"Data-prep failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
            raise RuntimeError("Data-prep script error.")
    finally:
        os.chdir(orig_cwd)

###############################################################################
# ---------------------- helper: run training ------------------------------- #
###############################################################################
def run_training_evaluation(
    *,
    keep_last_percent: float,
    navg: float,
    kernel: str,
    C: float,
    gamma: float,
    args
) -> dict:
    """
    Execute *run_model.py* in training mode (`--mode 0`).  Afterwards read the
    CSV written by that script with averaged validation metrics and return them
    as a dictionary.
    """
    cmd = [
        "python", args.train_script,
        "--mode",          "0",
        "--model_type",    args.model_type,
        "--data_scale",    args.data_scale,

        "--kernel", kernel,
        "--C",      str(C),
        "--gamma",  str(gamma),

        "--prefix",       args.prefix,
        "--data_dir",     args.data_dir,
        "--model_dir",    args.model_dir,
        "--output_file",  args.output_file,
        "--ref_id_col",   args.ref_id_col,
        "--ref_label_col", args.ref_label_col,
        "--scramble_fractions", str(SCR_FRACTION)
    ]
    if args.use_early_stopping: cmd.append("--use_early_stopping")

    logging.info(f"[Training] {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        logging.error(f"Training failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        raise RuntimeError("run_model.py error.")

    metrics_csv = f"final_metrics_{args.model_type}_trn_scr{FRAC_STR}.csv"
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"{metrics_csv} not found after training.")

    with open(metrics_csv) as f:
        row = next(csv.DictReader(f))

    return (
        {"MSE": float(row["MSE"]), "Pear": float(row["Pear"])}
        if args.model_type == "reg"
        else {"Accuracy": float(row["Accuracy"]), "MCC": float(row["MCC"])}
    )

###############################################################################
# ------------------------ Hyperopt objective ------------------------------- #
###############################################################################
def evaluate_hyperparams(params: dict, args) -> float:
    """
    Map Hyperopt's parameter dict → scalar loss.
    On any failure or NaN metric returns a large penalty (1e6).
    """
    keep_last = float(params["keep_last_percent"])
    navg      = float(params["navg"])
    kernel    = params["kernel"]
    C_val     = float(params["C"])
    gamma_val = float(params["gamma"])

    try:
        run_data_prep(keep_last, navg, args)
        metrics = run_training_evaluation(
            keep_last_percent = keep_last,
            navg              = navg,
            kernel            = kernel,
            C                 = C_val,
            gamma             = gamma_val,
            args              = args
        )
    except Exception as e:
        logging.error(f"[Objective] Trial failed: {e}")
        return 1e6

    # convert metrics → loss
    if args.model_type == "reg":
        mse, pear = metrics["MSE"], metrics["Pear"]
        if np.isnan(mse) or np.isnan(pear): return 1e6
        loss = 0.5 * (1.0 - pear) + 0.5 * mse
    else:
        acc, mcc = metrics["Accuracy"], metrics["MCC"]
        if np.isnan(acc) or np.isnan(mcc): return 1e6
        loss = 0.5 * (1.0 - acc) + 0.5 * (1.0 - mcc)

    logging.info(f"[Objective] params={params}, metrics={metrics}, loss={loss:.4f}")
    return loss

###############################################################################
# Main
###############################################################################
def main():
    args = parse_args()

    # Spark session for parallel Hyperopt trials
    spark = (SparkSession.builder
             .appName("SVMHyperopt")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    trials = SparkTrials(parallelism=args.spark_parallelism)

    # run Hyperopt
    best_raw = fmin(
        fn=lambda p: evaluate_hyperparams(p, args),
        space=search_space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        verbose=False
    )
    logging.info(f"Best raw indices: {best_raw}")

    # because kernel is hp.choice, translate index → string
    kernel_choices = ["linear", "rbf", "poly", "sigmoid"]
    best_kernel = kernel_choices[best_raw["kernel"]]

    best_cfg = {
        "keep_last_percent": float(best_raw["keep_last_percent"]),
        "navg":              float(best_raw["navg"]),
        "kernel":            best_kernel,
        "C":                 float(best_raw["C"]),
        "gamma":             float(best_raw["gamma"]),
        "model_type":        args.model_type,
        "data_scale":        args.data_scale
    }

    with open(f"best_hyperparams_{args.model_type}.json", "w") as f:
        json.dump(best_cfg, f, indent=4)
    logging.info("Best hyper-parameters saved.")

    # collect each trial's parameters and loss → CSV
    trial_rows = []
    for t in trials.trials:
        res  = t["result"]
        vals = t["misc"]["vals"]
        k_idx = vals["kernel"][0]

        trial_rows.append({
            "loss":   res.get("loss"),
            "status": res.get("status"),
            "keep_last_percent": vals["keep_last_percent"][0],
            "navg":              vals["navg"][0],
            "kernel":            kernel_choices[k_idx],
            "C":                 vals["C"][0],
            "gamma":             vals["gamma"][0]
        })

    pd.DataFrame(trial_rows).to_csv(f"hyperopt_results_{args.model_type}.csv", index=False)
    logging.info("All trial results saved.")

    spark.stop()


if __name__ == "__main__":
    main()
