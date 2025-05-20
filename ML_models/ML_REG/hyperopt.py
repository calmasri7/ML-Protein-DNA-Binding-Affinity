#!/usr/bin/env python
"""
hyperopt.py 

Hyperparameter optimisation wrapper specialised for strictly linear
models (i.e. `hidden_layers = 0`).  The workflow mirrors the previously
commented generic NN tuner but locks all architecture‐related search
hyperparameters to values that enforce no hidden layers, no dropout,
and no hidden units.

Sequence of operations for each Hyperopt trial

1. Data preparation  
   `torch_prep_kfold.py --process train` generates train/validation
   CSV splits (repeated K-fold, scramble fraction fixed at 0.0).
2. Training + evaluation  
   `run_model.py --mode 0` trains a strictly linear network
   (`--hidden_layers 0`) and writes a CSV summarising the averaged
   validation metrics across all folds/repeats.
3. Objective  
   Parse that CSV → convert to a scalar loss (lower = better).
4. SparkTrials  
   Run many trials in parallel via a local/cluster Spark session.

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
# Logging
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
    Build/parse CLI – identical to the generic tuner but with no hidden‐layer
    search (still keeps the arguments for completeness).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian hyperparameter optimisation for a strictly *linear* model.\n"
            "Delegates data prep to torch_prep_kfold.py and training to run_model.py."
        )
    )

    # External script paths
    parser.add_argument("--prep_script",  default="torch_prep_kfold.py")
    parser.add_argument("--train_script", default="run_model.py")

    # Dir / file paths
    parser.add_argument("--data_dir",  default="Data")
    parser.add_argument("--model_dir", default="Model")
    parser.add_argument("--reference_file", default="../Inputs/exp_data_all.csv")

    # Column names & feature CSVs
    parser.add_argument("--ref_id_col",    default="sequence")
    parser.add_argument("--ref_label_col", default="label")
    parser.add_argument("--filenames", nargs="+", required=True)
    parser.add_argument("--feature_id_col", default="sequence")
    parser.add_argument("--usecols", nargs="+")
    parser.add_argument("--keep_last_percent", type=float, default=0.0)

    # Misc
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_type", choices=["reg", "bin", "mclass"], required=True)
    parser.add_argument("--data_scale", choices=["log", "nonlog"], default="log")
    parser.add_argument("--use_early_stopping", action="store_true")

    # Spark / Hyperopt
    parser.add_argument("--spark_parallelism", type=int, default=4)
    parser.add_argument("--max_evals", type=int, default=50)

    # Repeated K-fold forwarding
    parser.add_argument("--kfold",       type=int, default=5)
    parser.add_argument("--num_repeats", type=int, default=5)

    # Default (fixed) NN args – still accepted but ignored by search
    parser.add_argument("--hidden_layers", type=int, default=0)
    parser.add_argument("--hidden_size",   type=int, default=0)
    parser.add_argument("--lrn_rate",      type=float, default=1e-4)
    parser.add_argument("--wt_decay",      type=float, default=1e-4)
    parser.add_argument("--dropout_input_output", type=float, default=0.0)
    parser.add_argument("--dropout_hidden",       type=float, default=0.0)
    parser.add_argument("--navg",          type=int, default=80)

    parser.add_argument("--prefix",      default="gbsa")
    parser.add_argument("--output_file", default="predictions")
    return parser.parse_args()

###############################################################################
# Constant scramble fraction for tuning
###############################################################################
SCR_FRACTION = 0.0
FRAC_STR = f"{SCR_FRACTION:.2f}".replace(".", "p")      # "0p00"

###############################################################################
# Hyperopt search space – only *data-prep* and *optimiser* parameters vary
###############################################################################
search_space = {
    'keep_last_percent':    hp.quniform('keep_last_percent', 5, 100, 5),
    'navg':                 hp.quniform('navg', 1, 400, 20),

    # architecture fixed to linear model
    'hidden_layers':        hp.choice('hidden_layers', [0]),      # always 0
    'hidden_size':          hp.quniform('hidden_size', 0, 0, 1),  # dummy 0

    # optimiser / regularisation
    'lrn_rate':             hp.loguniform('lrn_rate', np.log(1e-5), np.log(1e-2)),
    'wt_decay':             hp.loguniform('wt_decay', np.log(1e-10), np.log(1e-3)),

    # dropout fixed to 0
    'dropout_input_output': hp.uniform('dropout_input_output', 0.0, 0.0),
    'dropout_hidden':       hp.uniform('dropout_hidden',       0.0, 0.0),
}

###############################################################################
# ----------------- helper: run data-prep script ---------------------------- #
###############################################################################
def run_data_prep(keep_last_percent: float, navg: int, args):
    """
    Execute *torch_prep_kfold.py* inside `args.data_dir`.  Abort trial if
    script exits with non-zero status.
    """
    original_cwd = os.getcwd()
    os.chdir(args.data_dir)
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
        if args.filenames: cmd += ["--filenames"] + args.filenames
        if args.usecols:   cmd += ["--usecols"]   + args.usecols

        logging.info(f"[DataPrep] {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
        if res.returncode != 0:
            logging.error(f"Data-prep failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
            raise RuntimeError("Data-prep script failed.")
    finally:
        os.chdir(original_cwd)

###############################################################################
# ----------------- helper: run training script ----------------------------- #
###############################################################################
def run_training_evaluation(
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
    Execute *run_model.py* (training).  Expects that script writes
    `final_metrics_*_trn_scr0p00.csv` inside current directory.
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
        raise RuntimeError("Training script failed.")

    metrics_csv = f"final_metrics_{args.model_type}_trn_scr{FRAC_STR}.csv"
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"{metrics_csv} not produced.")

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
    Objective = failure penalty (1e6) OR weighted combination of metrics.
    """
    # unpack params from numpy→python types
    keep_last = float(params['keep_last_percent'])
    navg      = int(params['navg'])
    lr        = float(params['lrn_rate'])
    wd        = float(params['wt_decay'])

    try:
        run_data_prep(keep_last, navg, args)
        metrics = run_training_evaluation(
            keep_last_percent = keep_last,
            navg              = navg,
            hidden_size       = 0,
            hidden_layers     = 0,
            lrn_rate          = lr,
            wt_decay          = wd,
            dropout_input_output = 0.0,
            dropout_hidden    = 0.0,
            args = args
        )
    except Exception as e:
        logging.error(f"[Objective] Failure: {e}")
        return 1e6

    # scalar loss
    if args.model_type == "reg":
        mse, pear = metrics["MSE"], metrics["Pear"]
        if np.isnan(mse) or np.isnan(pear): return 1e6
        loss = 0.5 * mse + 0.5 * (1.0 - pear)
    else:
        acc, mcc = metrics["Accuracy"], metrics["MCC"]
        if np.isnan(acc) or np.isnan(mcc): return 1e6
        loss = 0.5 * (1.0 - acc) + 0.5 * (1.0 - mcc)

    logging.info(f"[Objective] params={params}, metrics={metrics}, loss={loss:.4f}")
    return loss

###############################################################################
# Main driver
###############################################################################
def main():
    args = parse_args()

    # Spark session for distributed trials
    spark = (SparkSession.builder
             .appName("HyperoptLinear")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())

    trials = SparkTrials(parallelism=args.spark_parallelism)

    # a *dummy* list to translate hidden_layers choice (always 0)
    hidden_choices = [0]

    best_raw = fmin(
        fn=lambda p: evaluate_hyperparams(p, args),
        space=search_space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        verbose=False
    )
    logging.info(f"Best raw indices: {best_raw}")

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

    with open(f"best_hyperparams_{args.model_type}.json", "w") as f:
        json.dump(best_cfg, f, indent=4)
    logging.info("Best hyper-parameters saved.")

    # collect trial results → CSV
    trial_rows = []
    for t in trials.trials:
        res  = t['result']
        vals = t['misc']['vals']
        trial_rows.append({
            "loss":   res.get('loss'),
            "status": res.get('status'),
            "keep_last_percent": vals['keep_last_percent'][0],
            "navg":              vals['navg'][0],
            "hidden_layers":     0,
            "hidden_size":       0,
            "lrn_rate":          vals['lrn_rate'][0],
            "wt_decay":          vals['wt_decay'][0]
        })
    pd.DataFrame(trial_rows).to_csv(f"hyperopt_results_{args.model_type}.csv", index=False)
    logging.info("All trial results saved.")

    spark.stop()


if __name__ == "__main__":
    main()
