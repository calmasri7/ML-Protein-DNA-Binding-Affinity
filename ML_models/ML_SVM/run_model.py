#!/usr/bin/env python

"""
run_model.py

This script trains or evaluates an SVM (SVR or SVC) using repeated K-fold splits
and optionally multiple scramble fractions. 

Modes:
------
1) mode=0 (training):
   - For each fraction in --scramble_fractions,
     - For each repeat_idx in [0..num_repeats-1] 
       and each fold_idx in [0..kfold-1]:
         * Load the CSV files:
             {prefix}_{model_type}_scrFRAC_trn_{repeat_idx}_{fold_idx}.csv
             {prefix}_{model_type}_scrFRAC_val_{repeat_idx}_{fold_idx}.csv
         * Train an SVR or SVC, save model to
             svm_fold_{repeat_idx}_{fold_idx}_{model_type}_scrFRAC.joblib
         * Evaluate on the validation split => gather metrics.

2) mode=1 (evaluation):
   - For each fraction in --scramble_fractions,
     - Load the final test file => {prefix}_{model_type}_scrFRAC_tst_final.csv
     - For each repeat_idx in [0..num_repeats-1], each fold_idx in [0..kfold-1],
       load the trained model => svm_fold_{repeat_idx}_{fold_idx}_{model_type}_scrFRAC.joblib
       Evaluate => gather metrics/predictions, then output final aggregated results.

Usage Examples:
---------------
- Training multiple fractions:
    python run_model_svm.py \
      --mode 0 \
      --model_type reg \
      --scramble_fractions 0.0 0.25 \
      --prefix gbsa \
      --kfold 5 \
      --num_repeats 5 \
      --kernel rbf \
      --C 1.0 \
      --gamma 0.1

- Evaluation on those same fractions:
    python run_model_svm.py \
      --mode 1 \
      --model_type reg \
      --scramble_fractions 0.0 0.25 \
      --prefix gbsa \
      --kfold 5 \
      --num_repeats 5 \
      --kernel rbf \
      --C 1.0 \
      --gamma 0.1
"""

import argparse
import sys
import os
import csv
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from collections import defaultdict, Counter
from scipy.stats import pearsonr
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    r2_score
)
from sklearn.svm import SVR, SVC
import joblib

###############################################################################
# Logging configuration
###############################################################################
logging.basicConfig(
    level=logging.DEBUG,                  # default verbosity
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# Utility helpers
###############################################################################
def fmt_float(x: float) -> str:
    """
    Format *x* to four decimal places.
    Returns the string `'NaN'` if *x* is not a float or is `np.nan`.
    """
    return f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "NaN"


def majority_vote(values: List[int]) -> int:
    """
    Return the most frequent integer in *values*.
    On ties `pandas.Series.mode()[0]` returns the *first* encountered mode.
    """
    return int(pd.Series(values).mode()[0])

###############################################################################
# Command-line interface
###############################################################################
def parse_args():
    """
    Build and parse CLI arguments for training / evaluation.
    """
    parser = argparse.ArgumentParser(
        description="SVM (SVR / SVC) with repeated K-fold CV and "
                    "optional scramble fractions."
    )

    # ------------------- operating mode ------------------------------- #
    parser.add_argument("--mode", type=int, default=0,
                        help="0 = training, 1 = evaluation on test set.")

    parser.add_argument("--model_type", type=str, default="reg",
                        choices=["reg", "bin", "mclass"],
                        help="'reg' → regression (SVR); "
                             "'bin' → binary SVC; "
                             "'mclass' → multi-class SVC.")

    # affects threshold used to binarize regression outputs
    parser.add_argument("--data_scale", type=str, default="log",
                        choices=["log", "nonlog"],
                        help="Used only when converting regression outputs "
                             "to classes for MCC / accuracy.")

    # ------------------- CV parameters -------------------------------- #
    parser.add_argument("--kfold", type=int, default=5,
                        help="#folds per repeat.")
    parser.add_argument("--num_repeats", type=int, default=5,
                        help="#repeated K-fold splits.")

    # ------------------- SVM hyper-parameters ------------------------- #
    parser.add_argument("--kernel", type=str, default="rbf",
                        help="'linear', 'rbf', 'poly', ...")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Penalty parameter C.")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Kernel coefficient (rbf/poly/sigmoid).")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="ε in ε-SVR (regression only).")

    # ------------------- file / dir paths ----------------------------- #
    parser.add_argument("--model_dir", type=str, default="Model/",
                        help="Directory to save *.joblib SVM files.")
    parser.add_argument("--data_dir", type=str, default="Data/",
                        help="Directory containing fold CSVs and test CSV.")
    parser.add_argument("--output_file", type=str, default="predictions",
                        help="Prefix for prediction/metric CSVs.")
    parser.add_argument("--prefix", type=str, default="gbsa",
                        help="Prefix used by data-prep script.")

    # ------------------- CSV column names ----------------------------- #
    parser.add_argument("--ref_id_col", type=str, default="sequence",
                        help="ID column (for aggregation).")
    parser.add_argument("--ref_label_col", type=str, default="label",
                        help="Target column.")

    # ------------------- label scrambling fractions ------------------- #
    parser.add_argument("--scramble_fractions", type=float, nargs="+",
                        default=[0.0],
                        help="Fractions used during data-prep; file names "
                             "contain e.g. 'scr0p25'.")
    return parser.parse_args()

###############################################################################
# Data I/O
###############################################################################
def load_csv_data(csv_file: str, args) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Read *csv_file* and return:
        * X  – features (float32 ndarray, shape [N, d])
        * y  – targets  (float32 ndarray, shape [N])
        * ids – list of sequence IDs (len = N)
    Raises if ID / label column missing.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Missing data file: {csv_file}")

    df = pd.read_csv(csv_file)
    id_col, lbl_col = args.ref_id_col, args.ref_label_col
    if id_col not in df.columns or lbl_col not in df.columns:
        raise ValueError(f"CSV {csv_file} must contain '{id_col}' and '{lbl_col}'.")

    feat_cols = [c for c in df.columns if c not in [id_col, lbl_col]]
    X   = df[feat_cols].values.astype(np.float32)
    y   = df[lbl_col].values.astype(np.float32)
    ids = df[id_col].tolist()
    return X, y, ids

###############################################################################
# SVM constructor
###############################################################################
def build_svm(args):
    """
    Instantiate and return:
        * SVR  if regression task
        * SVC  otherwise
    Hyper-parameters come from CLI.
    """
    if args.model_type == "reg":
        return SVR(kernel=args.kernel, C=args.C,
                   gamma=args.gamma, epsilon=args.epsilon)
    else:
        # binary / multi-class share same SVC call
        return SVC(kernel=args.kernel, C=args.C,
                   gamma=args.gamma, probability=True)

###############################################################################
# Training helper
###############################################################################
def train_svm_model(model, X_train, y_train, args):
    """
    Fit *model* on training data.
    Cast y→int for classification tasks (required by SVC).
    Returns the fitted model (in-place).
    """
    if args.model_type in ["bin", "mclass"]:
        y_train = y_train.astype(int)
    model.fit(X_train, y_train)
    return model

###############################################################################
# Evaluation routine
###############################################################################
def evaluate_svm_model(model, X, y, ids, args):
    """
    Run inference and compute metrics.

    Returns tuple:
        (mse, r2, pearson, mcc, accuracy, row_level_predictions)

    * For classification tasks (bin / mclass) → regression metrics are NaN.
    * For regression  → MCC / accuracy computed by thresholding predictions.
    """
    # aggregator groups rows belonging to the same ID
    aggregator = defaultdict(lambda: {"preds": [], "logits": [], "tgt": []})
    row_data   = []     # (id, pred, true) per row

    # ----------------------------------------------------------------- #
    # MULTI-CLASS CLASSIFICATION
    # ----------------------------------------------------------------- #
    if args.model_type == "mclass":
        logits = model.decision_function(X)           # shape [N, C]
        preds  = np.argmax(logits, axis=1)
        for i, uid in enumerate(ids):
            aggregator[uid]["logits"].append(logits[i])
            aggregator[uid]["tgt"].append(int(y[i]))
            row_data.append((uid, preds[i], y[i]))

        agg_preds, agg_tgts = [], []
        for d in aggregator.values():
            summed = np.sum(d["logits"], axis=0)      # vote by logit sum
            agg_preds.append(int(np.argmax(summed)))
            agg_tgts.append(majority_vote(d["tgt"]))

        mcc, acc = float('nan'), float('nan')
        if len(set(agg_tgts)) > 1:
            mcc  = matthews_corrcoef(agg_tgts, agg_preds)
            acc  = accuracy_score(agg_tgts, agg_preds)
        return (np.nan, np.nan, np.nan, mcc, acc, row_data)

    # ----------------------------------------------------------------- #
    # BINARY CLASSIFICATION
    # ----------------------------------------------------------------- #
    if args.model_type == "bin":
        preds = model.predict(X).astype(int)
        for i, uid in enumerate(ids):
            aggregator[uid]["preds"].append(int(preds[i]))
            aggregator[uid]["tgt"].append(int(y[i]))
            row_data.append((uid, preds[i], y[i]))

        agg_preds = [majority_vote(d["preds"]) for d in aggregator.values()]
        agg_tgts  = [majority_vote(d["tgt"])   for d in aggregator.values()]

        mcc, acc = float('nan'), float('nan')
        if len(set(agg_tgts)) > 1:
            mcc  = matthews_corrcoef(agg_tgts, agg_preds)
            acc  = accuracy_score(agg_tgts,  agg_preds)
        return (np.nan, np.nan, np.nan, mcc, acc, row_data)

    # ----------------------------------------------------------------- #
    # REGRESSION
    # ----------------------------------------------------------------- #
    preds = model.predict(X)
    for i, uid in enumerate(ids):
        aggregator[uid]["preds"].append(preds[i])
        aggregator[uid]["tgt"].append(y[i])
        row_data.append((uid, preds[i], y[i]))

    agg_preds = [np.mean(d["preds"]) for d in aggregator.values()]
    agg_tgts  = [np.mean(d["tgt"])   for d in aggregator.values()]

    mse = float(np.mean((np.array(agg_preds) - np.array(agg_tgts))**2))
    r2  = pear = float('nan')
    if len(agg_preds) > 1:
        r2   = float(r2_score(agg_tgts, agg_preds))
        pear,_ = pearsonr(agg_tgts, agg_preds)

    # additional classification-style metrics by thresholding ΔΔG
    thr = 0.0 if args.data_scale == "log" else 1.0
    pred_cls = (np.array(agg_preds) > thr).astype(int)
    tgt_cls  = (np.array(agg_tgts) > thr).astype(int)

    mcc = acc = float('nan')
    if len(set(tgt_cls)) > 1:
        mcc = matthews_corrcoef(tgt_cls, pred_cls)
        acc = accuracy_score(tgt_cls,  pred_cls)

    return (mse, r2, pear, mcc, acc, row_data)

###############################################################################
# CSV writer
###############################################################################
def save_predictions(predictions, filename: str):
    """
    Write per-row predictions to *filename* with columns:
        Label, Predicted, True
    """
    with open(filename, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Label", "Predicted", "True"])
        wr.writerows(predictions)

###############################################################################
# TRAINING MODE
###############################################################################
def run_training_mode(args):
    """
    Loop over scramble fractions → repeats → folds.
    Train an SVM for each split, save *.joblib, evaluate on validation split,
    and store metrics / row-level predictions.  
    Finally aggregate metrics & predictions across all splits.
    """
    os.makedirs(args.model_dir, exist_ok=True)

    for fraction in args.scramble_fractions:
        frac_str = f"{fraction:.2f}".replace(".", "p")
        logging.info(f"=== Training (scr{frac_str}) ===")

        all_metrics, all_predictions = [], []

        # iterate repeated K-fold indices
        for rep in range(args.num_repeats):
            for fold in range(args.kfold):
                trn_csv = os.path.join(
                    args.data_dir,
                    f"{args.prefix}_{args.model_type}_scr{frac_str}_trn_{rep}_{fold}.csv"
                )
                val_csv = os.path.join(
                    args.data_dir,
                    f"{args.prefix}_{args.model_type}_scr{frac_str}_val_{rep}_{fold}.csv"
                )
                if not (os.path.isfile(trn_csv) and os.path.isfile(val_csv)):
                    logging.warning(f"[scr{frac_str}] Missing CSV "
                                    f"(rep={rep}, fold={fold})")
                    continue

                # ------------- load data ------------------------------ #
                X_trn, y_trn, _   = load_csv_data(trn_csv, args)
                X_val, y_val, ids = load_csv_data(val_csv, args)

                # ------------- build / train -------------------------- #
                model = build_svm(args)
                logging.debug(f"Training SVM {model} "
                              f"(rep={rep}, fold={fold})")
                model = train_svm_model(model, X_trn, y_trn, args)

                # ------------- save model ----------------------------- #
                model_path = os.path.join(
                    args.model_dir,
                    f"svm_fold_{rep}_{fold}_{args.model_type}_scr{frac_str}.joblib"
                )
                joblib.dump(model, model_path)

                # ------------- validation metrics --------------------- #
                mse,r2,pear,mcc,acc,row_preds = evaluate_svm_model(
                    model, X_val, y_val, ids, args
                )
                all_metrics.append(
                    {"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc}
                )
                all_predictions.extend(row_preds)

                # save fold-level prediction CSV
                save_predictions(
                    row_preds,
                    f"{args.output_file}_{args.model_type}_scr{frac_str}"
                    f"_rep{rep}_fold{fold}.csv"
                )

                logging.info(f"[scr{frac_str} rep{rep} fold{fold}] "
                             f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, "
                             f"Pear={fmt_float(pear)}, MCC={fmt_float(mcc)}, "
                             f"Acc={fmt_float(acc)}")

        # -------- aggregate metrics across all splits --------------- #
        if all_metrics:
            avg = {}
            for key in all_metrics[0]:
                vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
                avg[key] = float(np.mean(vals)) if vals else float('nan')

            logging.info(f"[scr{frac_str}] FINAL avg validation metrics:")
            for k,v in avg.items():
                logging.info(f"  {k} = {fmt_float(v)}")

            with open(f"final_metrics_{args.model_type}_trn_scr{frac_str}.csv",
                      "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["MSE","R2","Pear","MCC","Accuracy"])
                w.writeheader()
                w.writerow({k: fmt_float(v) for k,v in avg.items()})

        # -------- aggregate predictions by ID ----------------------- #
        if all_predictions:
            agg = defaultdict(lambda: {"preds": [], "tgt": []})
            for uid, pred, tgt in all_predictions:
                agg[uid]["preds"].append(pred)
                agg[uid]["tgt"].append(tgt)

            labels, preds, tgts = [], [], []
            for uid, d in agg.items():
                labels.append(uid)
                if args.model_type in ["bin", "mclass"]:
                    preds.append(majority_vote(d["preds"]))
                    tgts .append(majority_vote(d["tgt"]))
                else:
                    preds.append(np.mean(d["preds"]))
                    tgts .append(np.mean(d["tgt"]))

            pd.DataFrame({
                "Label": labels,
                "AvgPredicted": preds,
                "AvgTrue": tgts
            }).to_csv(f"{args.output_file}_{args.model_type}_final_avg_scr{frac_str}.csv",
                      index=False)
            logging.info(f"[scr{frac_str}] Aggregated predictions saved.")

###############################################################################
# EVALUATION MODE
###############################################################################
def run_evaluation_mode(args):
    """
    Evaluate saved SVM checkpoints on the held-out test set and
    aggregate metrics / predictions across all repeats & folds.
    """
    os.makedirs(args.model_dir, exist_ok=True)

    for fraction in args.scramble_fractions:
        frac_str = f"{fraction:.2f}".replace(".", "p")
        test_csv = os.path.join(
            args.data_dir,
            f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_final.csv"
        )
        if not os.path.isfile(test_csv):
            logging.warning(f"Test CSV missing → {test_csv}")
            continue

        logging.info(f"=== Evaluation (scr{frac_str}) ===")
        X_tst, y_tst, ids_tst = load_csv_data(test_csv, args)
        if args.model_type in ["bin", "mclass"]:
            y_tst = y_tst.astype(int)

        preds_all, metrics_all = [], []

        for rep in range(args.num_repeats):
            for fold in range(args.kfold):
                model_path = os.path.join(
                    args.model_dir,
                    f"svm_fold_{rep}_{fold}_{args.model_type}_scr{frac_str}.joblib"
                )
                if not os.path.isfile(model_path):
                    logging.warning(f"Missing model {model_path}")
                    continue

                model = joblib.load(model_path)
                mse,r2,pear,mcc,acc,row_preds = evaluate_svm_model(
                    model, X_tst, y_tst, ids_tst, args
                )
                preds_all.extend(row_preds)
                metrics_all.append(
                    {"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc}
                )

                logging.info(f"[scr{frac_str} rep{rep} fold{fold}] "
                             f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, "
                             f"Pear={fmt_float(pear)}, MCC={fmt_float(mcc)}, "
                             f"Acc={fmt_float(acc)}")

        # -------- per-row predictions CSV ---------------------------- #
        save_predictions(
            preds_all,
            f"{args.output_file}_test_{args.model_type}_scr{frac_str}.csv"
        )

        # -------- average metrics across splits ---------------------- #
        if metrics_all:
            avg = {}
            for key in metrics_all[0]:
                vals = [m[key] for m in metrics_all if not np.isnan(m[key])]
                avg[key] = float(np.mean(vals)) if vals else float('nan')

            logging.info(f"[scr{frac_str}] FINAL test metrics:")
            for k,v in avg.items():
                logging.info(f"  {k} = {fmt_float(v)}")

            with open(f"final_metrics_{args.model_type}_tst_scr{frac_str}.csv",
                      "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["MSE","R2","Pear","MCC","Accuracy"])
                w.writeheader()
                w.writerow({k: fmt_float(v) for k,v in avg.items()})

###############################################################################
# Entry-point
###############################################################################
def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir,  exist_ok=True)

    logging.info(f"Scramble fractions   : {args.scramble_fractions}")
    logging.info(f"K-folds / repetions  : {args.kfold} / {args.num_repeats}")
    logging.info(f"Model type           : {args.model_type}")

    if   args.mode == 0:
        run_training_mode(args)
    elif args.mode == 1:
        run_evaluation_mode(args)
    else:
        logging.error("--mode must be 0 (train) or 1 (eval).")
        sys.exit(1)


if __name__ == "__main__":
    main()