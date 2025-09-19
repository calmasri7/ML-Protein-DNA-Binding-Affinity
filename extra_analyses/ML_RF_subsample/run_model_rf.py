#!/usr/bin/env python

"""
run_model_rf.py  ──  Random-Forest train/eval (scramble + subsample tags)
--------------------------------------------------------------------------

Two modes (aligned with files produced by torch_prep_kfold.py):

1) --mode 0  (TRAIN)
   • For each scramble fraction in --scramble_fractions
     and each subsample tag in --subsample_tags (e.g., sub100, sub75…):
       – Loop over repeats [0..num_repeats-1] and folds [0..kfold-1]
       – Load:
           Data/<prefix>_<mtype>_scr{frac}_<sub>_trn_{rep}_{fold}.csv
           Data/<prefix>_<mtype>_scr{frac}_<sub>_val_{rep}_{fold}.csv
       – Fit a RandomForest (reg or cls), save checkpoint:
           Model/rf_fold_{rep}_{fold}_{mtype}_scr{frac}_{sub}.pkl
       – Write fold-level predictions:
           <output_file>_{mtype}_scr{frac}_{sub}_rep{rep}_fold{fold}.csv
     • Aggregate metrics across all folds/repeats and write:
         final_metrics_{mtype}_trn_scr{frac}_{sub}.csv
     • Aggregate per-ID predictions (mean for reg; majority for cls):
         <output_file>_{mtype}_final_avg_scr{frac}_{sub}.csv

2) --mode 1  (EVAL)
   • For each scramble fraction (shared across all subtags):
       – Load test CSV (canonical preferred):
           Data/<prefix>_<mtype>_scr{frac}_tst_final.csv
         (Falls back to legacy: …_scr{frac}_{sub}_tst_final.csv if present.)
       – For each subsample tag + (rep, fold), load model checkpoint and score.
       – Write test predictions and average test metrics per (frac, sub):
           <output_file>_test_{mtype}_scr{frac}_{sub}.csv
           final_metrics_{mtype}_tst_scr{frac}_{sub}.csv

Notes
• Sequence-level aggregation:
  – Regression: mean prediction per ID; metrics on aggregated values.
  – Classification (bin/mclass): majority vote per ID (mclass uses prob-sum).
• “Sign” metrics for regression (MCC/Acc):
  – Threshold = 0.0 if --data_scale=log, else 1.0.
• Filenames rely on the same {scr} and {sub} tags produced by torch_prep_kfold.py.

Examples
--------
# Train RFs on all subtags (e.g., sub100, sub75, sub50)
python run_model_rf.py --mode 0 --model_type reg \
  --data_dir Data --model_dir Model --prefix gbsa \
  --kfold 5 --num_repeats 3 \
  --scramble_fractions 0.0 \
  --subsample_tags sub100 sub75 sub50 \
  --n_estimators 500 --max_depth None --max_features 0.5

# Evaluate on the (shared) clean test set
python run_model_rf.py --mode 1 --model_type reg \
  --data_dir Data --model_dir Model --prefix gbsa \
  --kfold 5 --num_repeats 3 \
  --scramble_fractions 0.0 \
  --subsample_tags sub100 sub75 sub50
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
    r2_score,
    mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

###############################################################################
# Logging Configuration
###############################################################################
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# Utility Functions
###############################################################################
def fmt_float(x: float) -> str:
    """Safely format a float to 4 decimals (NaN-safe for logging/CSV)."""
    return f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "NaN"

def majority_vote(values: List[int]) -> int:
    """Return the most frequent value (ties resolved by first mode)."""
    return int(pd.Series(values).mode()[0])

###############################################################################
# Argument Parsing
###############################################################################
def parse_args():
    """CLI args for RF training/evaluation with repeated K-fold and subsample tags."""
    parser = argparse.ArgumentParser(description="Random Forest training/evaluation with repeated K-fold splits.")
    parser.add_argument("--mode", type=int, default=0, help="Mode: 0 for training, 1 for evaluation.")
    parser.add_argument("--model_type", type=str, default="reg", choices=["reg", "bin", "mclass"],
                        help="Model type: 'reg', 'bin', or 'mclass'.")
    parser.add_argument("--model_obj", type=str, default="gbsa", help="Model/dataset descriptor (unused in filenames).")
    parser.add_argument("--data_scale", type=str, default="log", choices=["log", "nonlog"],
                        help="Affects sign-threshold for regression MCC/Accuracy.")
    parser.add_argument("--kfold", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--num_repeats", type=int, default=5, help="Repeated K-fold count.")
    parser.add_argument("--model_dir", type=str, default="Model/", help="Where RF .pkl checkpoints are saved/loaded.")
    parser.add_argument("--data_dir", type=str, default="Data/", help="Where CSV data files live.")
    parser.add_argument("--output_file", type=str, default="predictions",
                        help="Prefix for predictions/metrics CSV outputs.")
    parser.add_argument("--ref_id_col", type=str, default="sequence", help="ID column name in CSV.")
    parser.add_argument("--ref_label_col", type=str, default="label", help="Label column name in CSV.")

    # RF hyperparams
    parser.add_argument("--n_estimators", type=int, default=100, help="Trees in the forest.")
    parser.add_argument("--max_depth", type=str, default="None", help="Max tree depth ('None' for unlimited).")
    parser.add_argument("--max_features", type=str, default="auto", help="Features considered at split.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples to split.")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Min samples at leaf.")
    parser.add_argument("--random_state", type=int, default=42, help="RNG seed.")

    # Naming/tags
    parser.add_argument("--prefix", type=str, default="gbsa", help="Data filename prefix.")
    parser.add_argument("--scramble_fractions", type=float, nargs="+", default=[0.0],
                        help="Scramble fractions to match data files (e.g., 0.0 0.25 1.0).")
    parser.add_argument(
        "--subsample_tags", nargs="+", default=["sub100"],
        help=(
            "Subsample tags present in training/validation CSV filenames produced by torch_prep_kfold.py "
            "(e.g., sub100 sub75 sub50). Include sub100 (full data) if applicable."
        )
    )
    return parser.parse_args()

###############################################################################
# Data Loading
###############################################################################
def load_csv_data(csv_file: str, args) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Load a fold CSV: expects [ID, label, feature1, ...].
    Returns (X, y, ids) where ids is the per-row ID vector.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Data file not found: {csv_file}")
    df = pd.read_csv(csv_file, header=0)
    id_col = args.ref_id_col
    label_col = args.ref_label_col
    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV {csv_file} must contain '{id_col}' and '{label_col}'. Found: {df.columns.tolist()}")
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)
    id_list = df[id_col].tolist()
    return X, y, id_list

###############################################################################
# Build Random Forest
###############################################################################
def build_random_forest(args):
    """
    Instantiate RandomForestRegressor/Classifier with provided hyperparameters.
    """
    md = None if args.max_depth.lower() == "none" else int(args.max_depth)
    try:
        mf = float(args.max_features)
    except ValueError:
        mf = args.max_features
    if args.model_type == "reg":
        return RandomForestRegressor(
            n_estimators=args.n_estimators, max_depth=md, max_features=mf,
            min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state
        )
    else:
        return RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=md, max_features=mf,
            min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state
        )

###############################################################################
# Evaluate a Model
###############################################################################
def evaluate_model(rf, X_test, y_test, ids, args) -> Tuple[float, float, float, float, float, List[Tuple[Any, float, float]]]:
    """
    Sequence-level evaluation:
      • REG: predict per row → aggregate mean per ID → MSE, R2, Pearson.
              Also compute sign-based MCC/Acc using threshold from --data_scale.
      • BIN: predict class per row → majority per ID → MCC, Acc.
      • MCLASS: predict_proba per row → sum probabilities per ID → argmax → MCC, Acc.

    Returns: (MSE, R2, Pearson, MCC, Acc, row_tuples)
      where row_tuples = list of (ID, predicted, true) at the row level.
    """
    from collections import defaultdict
    if args.model_type == "mclass":
        probas = rf.predict_proba(X_test)
        y_int = y_test.astype(int)
        aggregator = defaultdict(lambda: {"probas": [], "tgt": []})
        row_data = []
        for i, uid in enumerate(ids):
            aggregator[uid]["probas"].append(probas[i])
            aggregator[uid]["tgt"].append(y_int[i])
            row_data.append((uid, int(np.argmax(probas[i])), int(y_int[i])))
        agg_preds, agg_tgts = [], []
        for uid, d in aggregator.items():
            sum_probs = np.sum(d["probas"], axis=0)
            pred_class = int(np.argmax(sum_probs))
            true_class = int(majority_vote(d["tgt"]))
            agg_preds.append(pred_class)
            agg_tgts.append(true_class)
        mcc_val, acc_val = float('nan'), float('nan')
        if len(set(agg_tgts)) > 1:
            mcc_val = matthews_corrcoef(agg_tgts, agg_preds)
            acc_val = accuracy_score(agg_tgts, agg_preds)
        return (np.nan, np.nan, np.nan, mcc_val, acc_val, row_data)
    elif args.model_type == "bin":
        preds = rf.predict(X_test)
        aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
        row_data = []
        for i, uid in enumerate(ids):
            aggregator[uid]["preds"].append(int(preds[i]))
            aggregator[uid]["tgt"].append(int(y_test[i]))
            row_data.append((uid, int(preds[i]), int(y_test[i])))
        agg_preds, agg_tgts = [], []
        for uid, d in aggregator.items():
            agg_preds.append(majority_vote(d["preds"]))
            agg_tgts.append(majority_vote(d["tgt"]))
        mcc_val, acc_val = float('nan'), float('nan')
        if len(set(agg_tgts)) > 1:
            mcc_val = matthews_corrcoef(agg_tgts, agg_preds)
            acc_val = accuracy_score(agg_tgts, agg_preds)
        return (np.nan, np.nan, np.nan, mcc_val, acc_val, row_data)
    else:
        preds = rf.predict(X_test)
        aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
        row_data = []
        for i, uid in enumerate(ids):
            aggregator[uid]["preds"].append(preds[i])
            aggregator[uid]["tgt"].append(y_test[i])
            row_data.append((uid, preds[i], y_test[i]))
        agg_preds, agg_tgts = [], []
        for uid, d in aggregator.items():
            agg_preds.append(np.mean(d["preds"]))
            agg_tgts.append(np.mean(d["tgt"]))
        mse_val = mean_squared_error(agg_tgts, agg_preds)
        r2_val, pear_val = float('nan'), float('nan')
        if len(agg_preds) > 1:
            r2_val = r2_score(agg_tgts, agg_preds)
            pear_val, _ = pearsonr(agg_tgts, agg_preds)
        thr = 0.0 if args.data_scale == "log" else 1.0
        pred_cls = (np.array(agg_preds) > thr).astype(int)
        tgt_cls = (np.array(agg_tgts) > thr).astype(int)
        mcc_val, acc_val = float('nan'), float('nan')
        if len(set(tgt_cls)) > 1:
            mcc_val = matthews_corrcoef(tgt_cls, pred_cls)
            acc_val = accuracy_score(tgt_cls, pred_cls)
        return (mse_val, r2_val, pear_val, mcc_val, acc_val, row_data)

###############################################################################
# Save Predictions to CSV
###############################################################################
def save_predictions(predictions, filename: str):
    """
    Write row-level predictions CSV with columns: [Label, Predicted, True].
    (Here 'Label' holds the ID/sequence identifier.)
    """
    with open(filename, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Label", "Predicted", "True"])
        wr.writerows(predictions)

###############################################################################
# TRAINING MODE: Repeated K-Fold across subtags
###############################################################################
def run_training_mode(args):
    """
    For each (scramble fraction, subsample tag):
      • Train RF per (repeat, fold) → save model + fold predictions
      • Aggregate metrics and per-ID predictions across all folds/repeats
    """
    os.makedirs(args.model_dir, exist_ok=True)
    for fraction in args.scramble_fractions:
        frac_str = f"{fraction:.2f}".replace(".", "p")
        for subtag in args.subsample_tags:          # e.g., sub100, sub75, sub50
            logging.info(f"=== Training: scr={fraction} | {subtag} ===")
            all_metrics, all_predictions = [], []   # reset for each subtag
            for repeat_idx in range(args.num_repeats):
                for fold_idx in range(args.kfold):
                    trn_file = os.path.join(
                        args.data_dir,
                        f"{args.prefix}_{args.model_type}_scr{frac_str}_{subtag}_trn_{repeat_idx}_{fold_idx}.csv"
                    )
                    val_file = os.path.join(
                        args.data_dir,
                        f"{args.prefix}_{args.model_type}_scr{frac_str}_{subtag}_val_{repeat_idx}_{fold_idx}.csv"
                    )

                    if not (os.path.isfile(trn_file) and os.path.isfile(val_file)):
                        logging.warning(f"Missing CSV for fraction={fraction}, repeat={repeat_idx}, fold={fold_idx}")
                        continue

                    X_train, y_train, ids_train = load_csv_data(trn_file, args)
                    X_val, y_val, ids_val = load_csv_data(val_file, args)
                    if args.model_type in ["bin", "mclass"]:
                        y_train = y_train.astype(int)
                        y_val = y_val.astype(int)

                    rf = build_random_forest(args)
                    logging.info(f"[Frac={fraction}] Repeat={repeat_idx}, Fold={fold_idx}: training model...")
                    rf.fit(X_train, y_train)

                    model_path = os.path.join(
                        args.model_dir,
                        f"rf_fold_{repeat_idx}_{fold_idx}_{args.model_type}_scr{frac_str}_{subtag}.pkl"
                    )
                    joblib.dump(rf, model_path)

                    mse, r2, pear, mcc, acc, fold_data = evaluate_model(rf, X_val, y_val, ids_val, args)
                    all_metrics.append({"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc})
                    all_predictions.extend(fold_data)

                    fold_pred_file = (
                        f"{args.output_file}_{args.model_type}_scr{frac_str}_{subtag}"
                        f"_rep{repeat_idx}_fold{fold_idx}.csv"
                    )
                    save_predictions(fold_data, fold_pred_file)
                    logging.info(
                        f"[Frac={fraction}, Rep={repeat_idx}, Fold={fold_idx}] "
                        f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, Pear={fmt_float(pear)}, "
                        f"MCC={fmt_float(mcc)}, Acc={fmt_float(acc)}"
                    )

            # Average metrics across all folds/repeats for this (scr, subtag)
            if all_metrics:
                keys = all_metrics[0].keys()
                avg_metrics = {k: float(np.mean([m[k] for m in all_metrics if not np.isnan(m[k])])) for k in keys}
                logging.info(f"[Frac={fraction}] Final Average Training Metrics (all repeats/folds):")
                for k, v in avg_metrics.items():
                    logging.info(f"  {k} = {fmt_float(v)}")
                final_csv = f"final_metrics_{args.model_type}_trn_scr{frac_str}_{subtag}.csv"
                with open(final_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["MSE", "R2", "Pear", "MCC", "Accuracy"])
                    w.writeheader()
                    w.writerow({k: fmt_float(avg_metrics[k]) for k in ["MSE", "R2", "Pear", "MCC", "Accuracy"]})

            # Aggregate per-ID predictions (mean or majority) across all folds/repeats
            if all_predictions:
                aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
                for uid, pred, tgt in all_predictions:
                    aggregator[uid]["preds"].append(pred)
                    aggregator[uid]["tgt"].append(tgt)
                final_labels, final_preds, final_tgts = [], []
                for lbl, d in aggregator.items():
                    final_labels.append(lbl)
                    if args.model_type in ["bin", "mclass"]:
                        final_preds.append(majority_vote(d["preds"]))
                        final_tgts.append(majority_vote(d["tgt"]))
                    else:
                        final_preds.append(np.mean(d["preds"]))
                        final_tgts.append(np.mean(d["tgt"]))
                agg_df = pd.DataFrame({
                    "Label": final_labels,        # sequence ID
                    "AvgPredicted": final_preds,  # per-ID aggregated prediction
                    "AvgTrue": final_tgts         # per-ID aggregated target
                })
                agg_csv = f"{args.output_file}_{args.model_type}_final_avg_scr{frac_str}_{subtag}.csv"
                agg_df.to_csv(agg_csv, index=False)
                logging.info(f"Final aggregated predictions saved: {agg_csv}")

###############################################################################
# EVALUATION MODE
###############################################################################
def _resolve_test_csv(prefix: str, model_type: str, frac_str: str, subtag: str, data_dir: str) -> str:
    """
    Determine test CSV path:
      • Prefer canonical (no subtag): <prefix>_<mtype>_scr{frac}_tst_final.csv
      • Fallback to legacy with subtag if present.
    """
    canonical = os.path.join(data_dir, f"{prefix}_{model_type}_scr{frac_str}_tst_final.csv")
    legacy    = os.path.join(data_dir, f"{prefix}_{model_type}_scr{frac_str}_{subtag}_tst_final.csv")
    if os.path.isfile(canonical):
        return canonical
    if os.path.isfile(legacy):
        logging.warning(f"Using legacy test file with subtag: {os.path.basename(legacy)}")
        return legacy
    return canonical  # will fail later with a clear error


def run_evaluation_mode(args):
    """
    Evaluate checkpoints on the shared test set (per scramble fraction):
      • Load test CSV once per fraction
      • Loop over subtags and (repeat, fold) models
      • Save row-level predictions + average metrics per (scr, subtag)
    """
    os.makedirs(args.model_dir, exist_ok=True)

    for fraction in args.scramble_fractions:
        frac_str = f"{fraction:.2f}".replace(".", "p")

        # Load the (shared) test set once per scramble fraction
        test_csv = _resolve_test_csv(args.prefix, args.model_type, frac_str, args.subsample_tags[0], args.data_dir)
        if not os.path.isfile(test_csv):
            logging.warning(f"[scr={fraction}] test CSV not found → {test_csv}")
            continue

        logging.info(f"=== EVAL  scr={fraction}  test={os.path.basename(test_csv)} ===")
        X_test, y_test, ids_test = load_csv_data(test_csv, args)
        if args.model_type in ["bin", "mclass"]:
            y_test = y_test.astype(int)

        # Evaluate each subtag’s models on the same test set
        for subtag in args.subsample_tags:
            predictions = []
            test_metrics = []

            for repeat_idx in range(args.num_repeats):
                for fold_idx in range(args.kfold):
                    model_path = os.path.join(
                        args.model_dir,
                        f"rf_fold_{repeat_idx}_{fold_idx}_{args.model_type}_scr{frac_str}_{subtag}.pkl"
                    )
                    if not os.path.isfile(model_path):
                        logging.debug(f"  ↳ missing model  rep={repeat_idx}  fold={fold_idx}  sub={subtag}")
                        continue

                    rf = joblib.load(model_path)
                    mse, r2, pear, mcc, acc, fold_preds = evaluate_model(
                        rf, X_test, y_test, ids_test, args
                    )
                    predictions.extend(fold_preds)
                    test_metrics.append({"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc})
                    logging.info(
                        f"[scr={fraction} sub={subtag} rep={repeat_idx} fold={fold_idx}] "
                        f"MSE={fmt_float(mse)} R2={fmt_float(r2)} Pear={fmt_float(pear)} "
                        f"MCC={fmt_float(mcc)} Acc={fmt_float(acc)}"
                    )

            if predictions:
                pred_csv = f"{args.output_file}_test_{args.model_type}_scr{frac_str}_{subtag}.csv"
                save_predictions(predictions, pred_csv)
                logging.info(f"[scr={fraction} | {subtag}] predictions → {pred_csv}")

            if test_metrics:
                keys = list(test_metrics[0].keys())
                avg = {k: float(np.nanmean([m[k] for m in test_metrics])) for k in keys}
                logging.info(f"[scr={fraction} | {subtag}] AVG test metrics: " +
                             " ".join([f"{k}={fmt_float(v)}" for k, v in avg.items()]))
                agg_csv = f"final_metrics_{args.model_type}_tst_scr{frac_str}_{subtag}.csv"
                with open(agg_csv, "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=keys)
                    wr.writeheader()
                    wr.writerow({k: fmt_float(avg[k]) for k in keys})
            else:
                logging.warning(f"[scr={fraction} | {subtag}] no models found; nothing written.")

###############################################################################
# Main Entry Point
###############################################################################
def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    logging.info(f"Scramble fractions: {args.scramble_fractions}")
    logging.info(f"K-fold={args.kfold}, num_repeats={args.num_repeats}")
    if args.mode == 0:
        run_training_mode(args)
    elif args.mode == 1:
        run_evaluation_mode(args)
    else:
        logging.error("Mode must be 0 or 1.")
        sys.exit(1)

if __name__ == "__main__":
    main()
