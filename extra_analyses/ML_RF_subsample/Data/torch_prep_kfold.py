#!/usr/bin/env python

"""
torch_prep_kfold.py  ──  data-set builder with *subsampling* support
--------------------------------------------------------------------

Three entry points (exactly as before)

1) --initial_split
   • Merge reference + features on a shared ID.
   • Optional *sequence-level* scrambling on the TRAIN split only
     (first value of --scramble_fractions is used).
   • Split by unique sequence IDs (stratified-group for classification).
   • Write:
       gbsa_<mtype>_scrXX_trn_final.csv
       gbsa_<mtype>_scrXX_tst_preprocess.csv

2) --process train
   • (Optional) keep-last-% of rows per sequence.
   • Average frames per sequence in chunks of --navg.
   • Compute μ,σ on the *clean averaged train*; save as
       gbsa_<mtype>_train_stats.csv
   • Standardize features with those stats (labels untouched).
   • For each subsample fraction in --subsample_fracs (e.g., 1.0, 0.75, 0.50):
       – Retain that fraction of UNIQUE sequences (sequence-level subsampling).
       – Build repeated K-fold CSVs per fraction.

   Filenames carry a subsample tag:  «sub50»  (=50 % of sequences kept).

3) --process test
   • Apply the same keep-last-% and averaging.
   • Standardize using the *stored train stats* (no subsampling, no extra scrambling).
   • Write:
       gbsa_<mtype>_scrXX_tst_final.csv

---------------------------------------------------------------------------
New command-line flag
---------------------------------------------------------------------------
--subsample_fracs   1.0 0.75 0.50 ...
  Fraction(s) of UNIQUE sequences to keep for training-fold generation.
  Include 1.0 to preserve the full-data condition.

---------------------------------------------------------------------------
File naming scheme (train/val folds)
---------------------------------------------------------------------------
gbsa_<mtype>_scr0p00_sub50_trn_0_2.csv
gbsa_<mtype>_scr0p00_sub50_val_0_2.csv
     ^            ^      ^      ^ ^
     |            |      |      | └─ fold index
     |            |      |      └──── repeat index
     |            |      └─────────── subsample tag (sub100, sub75, sub50, …)
     |            └────────────────── scramble fraction (unchanged)
     └─────────────────────────────── model-type (reg / bin / mclass)

Notes
• Standardization stats (μ,σ) are computed ONCE from the averaged, full train split
  and reused for all subsample fractions (fair comparison).
• Subsampling is sequence-level: keep/remove entire sequences, not individual rows.
• Classification uses StratifiedGroupKFold; regression uses GroupKFold.
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Any, Optional, List, Tuple
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import os

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# Argument parsing
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for data preparation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Data preparation script with three modes:\n"
            "1) --initial_split : Merge data and do train/test split (with optional sequence-level scrambling).\n"
            "2) --process train : Process the training set (filter, average, standardize, repeated K-fold, subsampling).\n"
            "3) --process test  : Process the test set (filter, average, standardize)."
        )
    )
    # Input references
    parser.add_argument("--reference_file", type=str,
                        help="Path to the reference CSV, used if --initial_split.")
    parser.add_argument("--ref_id_col", type=str, default="sequence",
                        help="Name of the ID column in the reference CSV.")
    parser.add_argument("--ref_label_col", type=str, default="label",
                        help="Name of the label column in the reference CSV.")

    # Subsampling control (NEW)
    parser.add_argument(
        "--subsample_fracs", type=float, nargs="+", default=[1.0],
        help=(
            "Fractions of UNIQUE sequences to retain during --process train.\n"
            "Example: 1.0 0.75 0.50 0.25 will create four training-set variants.\n"
            "Include 1.0 to keep the full-size split as a baseline."
        )
    )

    # Feature data
    parser.add_argument("--filenames", nargs="+",
                        help="CSV files containing features (used if --initial_split).")
    parser.add_argument("--feature_id_col", type=str, default="sequence",
                        help="Name of the ID column in the feature CSV(s).")
    parser.add_argument("--usecols", type=str, nargs="+", default=None,
                        help="Optional columns to load from feature CSV(s). Must include ID col if needed.")

    # Data splits / randomization
    parser.add_argument("--keep_last_percent", type=float, default=0.0,
                        help="Keep only the last X%% of rows per sequence. If <=0 or >=100, keep all.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for shuffling/splitting.")
    parser.add_argument("--model_type", type=str, choices=["reg", "bin", "mclass"], required=True,
                        help="Model type: 'reg', 'bin', or 'mclass'.")
    parser.add_argument("--navg", type=int, default=50,
                        help="Number of entries to average together per sequence chunk.")
    parser.add_argument("--kfold", type=int, default=5,
                        help="Number of cross-validation folds.")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Number of times to repeat the K-fold split for the training set.")
    parser.add_argument("--test_percentage", type=float, default=0.15,
                        help="Fraction of dataset to keep as test set.")
    parser.add_argument("--prefix", type=str, default="gbsa",
                        help="Prefix for output files (e.g. 'gbsa_reg_tst_final.csv').")

    # Script mode
    parser.add_argument("--initial_split", action="store_true",
                        help="Perform the initial train/test split on merged data, then exit.")
    parser.add_argument("--process", choices=["train", "test"],
                        help="Process the final train or test set (not used with --initial_split).")

    # Sequence-level scrambling fraction
    parser.add_argument("--scramble_fractions", type=float, nargs="+", default=[0.0],
                        help=(
                            "List of scramble fractions. Only the first value is used. "
                            "Sequence-level scrambling means that fraction of sequences in the training set "
                            "will have their label overwritten by another sequence's label."
                        ))

    return parser.parse_args()

###############################################################################
# Data loading (Reference + Features)
###############################################################################
def load_reference_data(filename: str, id_col: str, label_col: str) -> pd.DataFrame:
    """
    Load a reference CSV containing at least an ID column and a label column.
    """
    df = pd.read_csv(filename)
    if id_col not in df.columns or label_col not in df.columns:
        logging.error(f"Reference file must contain '{id_col}' and '{label_col}'. "
                      f"Found columns: {df.columns.tolist()}")
        sys.exit(1)
    return df[[id_col, label_col]].copy()

def load_feature_data(filename: str, id_col: str, usecols: Optional[List[str]]) -> pd.DataFrame:
    """
    Load a feature CSV file. If usecols is provided, only those columns are read.
    Must contain 'id_col'.
    """
    if usecols:
        df = pd.read_csv(filename, usecols=usecols)
    else:
        df = pd.read_csv(filename)
    if id_col not in df.columns:
        logging.error(f"Feature file {filename} missing '{id_col}'. Columns: {df.columns.tolist()}")
        sys.exit(1)
    return df

###############################################################################
# Sequence-level scrambling
###############################################################################
def scramble_sequences(df: pd.DataFrame, id_col: str, label_col: str, frac: float, seed: int) -> pd.DataFrame:
    """
    Copy the label of a *different* random sequence into `frac` of sequences (entire sequence overwritten).
    If frac is too small to affect at least one sequence, return unchanged.
    """
    if frac <= 0.0:
        return df

    df_out = df.copy()
    unique_seqs = df_out[id_col].unique()
    n_to_scramble = int(len(unique_seqs) * frac)
    if n_to_scramble < 1:
        return df_out

    rng = np.random.default_rng(seed)
    scramble_seqs = rng.choice(unique_seqs, size=n_to_scramble, replace=False)

    for seq in scramble_seqs:
        # pick a different sequence as the donor
        donors = [s for s in unique_seqs if s != seq]
        if not donors:
            continue
        donor = rng.choice(donors)
        donor_label = df_out.loc[df_out[id_col] == donor, label_col].iloc[0]
        df_out.loc[df_out[id_col] == seq, label_col] = donor_label

    return df_out

###############################################################################
# Filtering, averaging, and standardization
###############################################################################
def keep_last_n_percent(df: pd.DataFrame, seq_col: str, keep_percent: float) -> pd.DataFrame:
    """
    Retain only the last `keep_percent`% of rows in each sequence group.
    Sort by 'run' if present to preserve trajectory order; otherwise keep current order.
    """
    if keep_percent <= 0 or keep_percent >= 100:
        return df
    if "run" in df.columns:
        df_sorted = df.sort_values([seq_col, "run"], kind="mergesort")
    else:
        df_sorted = df.copy()
    group_sizes = df_sorted.groupby(seq_col)[seq_col].transform("size")
    cumcount = df_sorted.groupby(seq_col).cumcount()
    n_keep = (group_sizes * (keep_percent / 100.0)).astype(int).mask(lambda s: s < 1, 1)
    mask = cumcount >= (group_sizes - n_keep)
    return df_sorted[mask].reset_index(drop=True)

def average_features_for_sequence(
    df: pd.DataFrame,
    navg: int,
    id_col: str,
    label_col: str,
    random_state: int
) -> pd.DataFrame:
    """
    Shuffle the sequence's rows, chunk into size `navg`, and average numeric features.
    Label is taken from the first row of each chunk (assumed consistent within a sequence).
    """
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [id_col, label_col]]

    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_chunks = len(df_shuffled) // navg
    if n_chunks < 1:
        return pd.DataFrame(columns=df.columns)

    for i in range(n_chunks):
        chunk = df_shuffled.iloc[i * navg : (i + 1) * navg]
        row_dict = {col: chunk[col].mean() for col in feature_cols}
        row_dict[label_col] = chunk[label_col].iloc[0]
        row_dict[id_col] = chunk[id_col].iloc[0]
        results.append(row_dict)

    df_out = pd.DataFrame(results)
    col_order = [id_col] + sorted([c for c in df_out.columns if c != id_col])
    return df_out[col_order]

def average_features_for_mutants(
    df: pd.DataFrame,
    navg: int,
    id_col: str,
    label_col: str,
    random_state: int
) -> pd.DataFrame:
    """
    Apply averaging per sequence group and concatenate.
    """
    all_chunks = []
    for _, group in df.groupby(id_col):
        chunk_df = average_features_for_sequence(group, navg, id_col, label_col, random_state)
        all_chunks.append(chunk_df)
    if not all_chunks:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(all_chunks, ignore_index=True)

def compute_mean_std(
    df: pd.DataFrame,
    model_type: str,
    id_col: str,
    label_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean/std for numeric feature columns (excluding ID and label).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in (id_col, label_col):
        if col in numeric_cols:
            numeric_cols.remove(col)
    means = [(c, df[c].mean()) for c in numeric_cols]
    stds  = [(c, df[c].std())  for c in numeric_cols]
    return (
        pd.DataFrame(means, columns=["colname", "mean"]),
        pd.DataFrame(stds,  columns=["colname", "std"])
    )

def save_mean_std(mean_df: pd.DataFrame, std_df: pd.DataFrame, filename: str) -> None:
    """
    Save μ,σ in a single CSV with columns: colname, mean, std.
    """
    merged = pd.merge(mean_df, std_df, on="colname")
    merged.to_csv(filename, index=False)

def load_mean_std(filename: str) -> pd.DataFrame:
    """
    Load μ,σ CSV and validate required columns.
    """
    if not os.path.isfile(filename):
        logging.error(f"Mean/Std file not found: {filename}")
        sys.exit(1)
    df_stats = pd.read_csv(filename)
    needed = {"colname", "mean", "std"}
    if not needed.issubset(df_stats.columns):
        logging.error(f"Mean/Std file missing columns. Found: {df_stats.columns.tolist()}")
        sys.exit(1)
    return df_stats

def apply_standardization(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    model_type: str,
    id_col: str,
    label_col: str
) -> pd.DataFrame:
    """
    Standardize features using provided μ,σ (zero-variance columns are mean-centered).
    """
    df_std = df.copy()
    means = dict(zip(stats_df["colname"], stats_df["mean"]))
    stds  = dict(zip(stats_df["colname"], stats_df["std"]))
    for col in df_std.columns:
        if col in (id_col, label_col):
            continue
        if col in means and col in stds:
            mu, sigma = means[col], stds[col]
            if sigma == 0 or np.isnan(sigma):
                df_std[col] = df_std[col] - mu
            else:
                df_std[col] = (df_std[col] - mu) / sigma
    return df_std

###############################################################################
# Main
###############################################################################
def main():
    args = parse_arguments()

    # Only the first scramble fraction is used (kept for compatibility with other scripts).
    scr_frac = args.scramble_fractions[0]
    frac_str = f"{scr_frac:.2f}".replace(".", "p")

    # ─────────────────────────── 1) INITIAL SPLIT ───────────────────────────
    if args.initial_split:
        if not (args.reference_file and args.filenames):
            logging.error("Must provide --reference_file and --filenames when using --initial_split.")
            sys.exit(1)

        # Load reference (ID + label)
        df_ref = load_reference_data(args.reference_file, args.ref_id_col, args.ref_label_col)
        logging.info(f"Loaded reference data: shape={df_ref.shape}")

        # Load and concatenate feature CSVs
        feats = [load_feature_data(f, args.feature_id_col, args.usecols) for f in args.filenames]
        df_feat = pd.concat(feats, ignore_index=True)
        logging.info(f"Combined feature data => shape={df_feat.shape}")

        # Align ID column names and merge
        if args.feature_id_col != args.ref_id_col:
            df_feat = df_feat.rename(columns={args.feature_id_col: args.ref_id_col})
        df = pd.merge(df_feat, df_ref, on=args.ref_id_col, how="inner")
        logging.info(f"After merging => shape={df.shape}")

        # Shuffle for reproducibility
        df = df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

        # Split by unique sequence IDs (stratified for classification)
        if args.model_type in ["bin", "mclass"]:
            y = df[args.ref_label_col]
            groups = df[args.ref_id_col]
            gkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.random_state)
            train_idx, test_idx = next(gkf.split(df, y, groups=groups))
            df_tr, df_te = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        else:
            seqs = df[args.ref_id_col].unique()
            np.random.seed(args.random_state)
            np.random.shuffle(seqs)
            n_tr = int((1 - args.test_percentage) * len(seqs))
            tr_seqs, te_seqs = seqs[:n_tr], seqs[n_tr:]
            df_tr = df[df[args.ref_id_col].isin(tr_seqs)].copy()
            df_te = df[df[args.ref_id_col].isin(te_seqs)].copy()

        # Optional sequence-level scrambling on TRAIN only
        if scr_frac > 0:
            df_tr = scramble_sequences(df_tr, args.ref_id_col, args.ref_label_col, scr_frac, args.random_state)

        # Drop 'run' if present (downstream processing reorders anyway)
        for d in (df_tr, df_te):
            if "run" in d.columns:
                d.drop(columns=["run"], inplace=True, errors="ignore")

        # Save
        train_file = f"{args.prefix}_{args.model_type}_scr{frac_str}_trn_final.csv"
        test_file  = f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_preprocess.csv"
        df_tr.to_csv(train_file, index=False)
        df_te.to_csv(test_file,  index=False)
        logging.info(f"Scr={scr_frac}: train => {train_file}, shape={df_tr.shape}")
        logging.info(f"Scr={scr_frac}: test  => {test_file},  shape={df_te.shape}")
        logging.info("Initial split completed. Exiting.")
        sys.exit(0)

    # ─────────────────────────── 2) PROCESS MODE ────────────────────────────
    if args.process is None:
        logging.error("Must specify either --initial_split or --process [train/test].")
        sys.exit(1)

    if args.process == "train":
        # Load saved TRAIN CSV
        train_file = f"{args.prefix}_{args.model_type}_scr{frac_str}_trn_final.csv"
        if not os.path.isfile(train_file):
            logging.error(f"Training file not found => {train_file}")
            sys.exit(1)

        df_train = pd.read_csv(train_file)
        logging.info(f"Loaded training CSV => {train_file}, shape={df_train.shape}")

        if "run" in df_train.columns:
            df_train.drop(columns=["run"], inplace=True, errors="ignore")

        # (Optional) keep last X% of rows per sequence
        df_train = keep_last_n_percent(df_train, args.ref_id_col, args.keep_last_percent)

        # Average numeric features per sequence in chunks of --navg
        df_train_avg = average_features_for_mutants(
            df_train, navg=args.navg, id_col=args.ref_id_col,
            label_col=args.ref_label_col, random_state=args.random_state
        )
        logging.info(f"After averaging => shape={df_train_avg.shape}")

        # Compute and save μ,σ on the averaged, full train
        mean_df, std_df = compute_mean_std(df_train_avg, args.model_type, args.ref_id_col, args.ref_label_col)
        stats_file = f"{args.prefix}_{args.model_type}_train_stats.csv"
        save_mean_std(mean_df, std_df, stats_file)
        logging.info(f"Saved training mean/std => {stats_file}")

        # Standardize using saved μ,σ
        df_train_std = apply_standardization(
            df_train_avg, load_mean_std(stats_file),
            args.model_type, args.ref_id_col, args.ref_label_col
        )
        # Shuffle once for downstream splitting
        df_train_std = df_train_std.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

        # Iterate over subsample fractions (sequence-level retention)
        sub_fracs = sorted(set(args.subsample_fracs), reverse=True)   # e.g., [1.0, 0.75, 0.5]
        for sf in sub_fracs:
            frac_tag = f"sub{int(sf*100):02d}"        # 1.0 → sub100, 0.50 → sub50, …
            if sf < 1.0:
                rng = np.random.default_rng(args.random_state + int(sf*1000))
                uniq = df_train_std[args.ref_id_col].unique()
                n_keep = max(int(len(uniq) * sf), args.kfold)  # keep ≥ kfold sequences
                keep = rng.choice(uniq, size=n_keep, replace=False)
                df_work = df_train_std[df_train_std[args.ref_id_col].isin(keep)].copy()
            else:
                df_work = df_train_std.copy()

            logging.info(f"[{frac_tag}] after subsampling => shape={df_work.shape}")

            # Select splitter
            if args.model_type in ["bin", "mclass"]:
                splitter = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=args.random_state)
                split_iter = splitter.split(df_work, df_work[args.ref_label_col], groups=df_work[args.ref_id_col])
            else:
                splitter = GroupKFold(n_splits=args.kfold)
                split_iter = splitter.split(df_work, groups=df_work[args.ref_id_col])

            # Repeated K-fold
            for repeat_idx in range(args.num_repeats):
                repeat_seed = args.random_state + 100 * repeat_idx

                # For classification, re-instantiate stratified splitter per repeat
                if args.model_type in ["bin", "mclass"]:
                    splitter = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=repeat_seed)
                    split_iter = splitter.split(df_work, df_work[args.ref_label_col], groups=df_work[args.ref_id_col])
                else:
                    # GroupKFold is deterministic; shuffle sequence order for variety
                    seq_ids = df_work[args.ref_id_col].unique()
                    rng = np.random.default_rng(repeat_seed)
                    rng.shuffle(seq_ids)
                    idx_map = {s: i for i, s in enumerate(seq_ids)}
                    df_work["_tmp_order"] = df_work[args.ref_id_col].map(idx_map)
                    df_work = df_work.sort_values("_tmp_order")
                    split_iter = GroupKFold(n_splits=args.kfold).split(df_work, groups=df_work[args.ref_id_col])
                    df_work.drop(columns="_tmp_order", inplace=True)

                for fold_idx, (trn_idx, val_idx) in enumerate(split_iter):
                    df_fold_trn = df_work.iloc[trn_idx].copy()
                    df_fold_val = df_work.iloc[val_idx].copy()

                    # Put ID first for readability/compat
                    col_order = [args.ref_id_col] + [c for c in df_fold_trn.columns if c != args.ref_id_col]
                    df_fold_trn = df_fold_trn[col_order]
                    df_fold_val = df_fold_val[col_order]

                    # Subsample tag in filenames
                    trg = f"{args.prefix}_{args.model_type}_scr{frac_str}_{frac_tag}_trn_{repeat_idx}_{fold_idx}.csv"
                    val = f"{args.prefix}_{args.model_type}_scr{frac_str}_{frac_tag}_val_{repeat_idx}_{fold_idx}.csv"
                    df_fold_trn.to_csv(trg, index=False)
                    df_fold_val.to_csv(val, index=False)
                    logging.info(f"[{frac_tag}] Rep={repeat_idx}, Fold={fold_idx}: "
                                 f"Train shape={df_fold_trn.shape}, Val shape={df_fold_val.shape}")

    elif args.process == "test":
        # Load saved TEST CSV (preprocessed, no noise/subsample)
        test_file = f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_preprocess.csv"
        if not os.path.isfile(test_file):
            logging.error(f"Test file not found => {test_file}")
            sys.exit(1)

        df_test = pd.read_csv(test_file)
        logging.info(f"Loaded test CSV => {test_file}, shape={df_test.shape}")

        if "run" in df_test.columns:
            df_test.drop(columns=["run"], inplace=True, errors="ignore")

        # Same filtering + averaging as train
        df_test = keep_last_n_percent(df_test, args.ref_id_col, args.keep_last_percent)
        df_test_avg = average_features_for_mutants(
            df_test, navg=args.navg, id_col=args.ref_id_col,
            label_col=args.ref_label_col, random_state=args.random_state
        )
        logging.info(f"After averaging => shape={df_test_avg.shape}")

        # Standardize using train stats
        stats_file = f"{args.prefix}_{args.model_type}_train_stats.csv"
        if not os.path.isfile(stats_file):
            logging.error(f"Train stats not found => {stats_file}")
            sys.exit(1)

        df_test_std = apply_standardization(
            df_test_avg, load_mean_std(stats_file),
            args.model_type, args.ref_id_col, args.ref_label_col
        )
        df_test_std = df_test_std.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

        final_test_csv = f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_final.csv"
        df_test_std.to_csv(final_test_csv, index=False)
        logging.info(f"Final test CSV saved: {final_test_csv}, shape={df_test_std.shape}")
        sys.exit(0)


if __name__ == "__main__":
    main()
