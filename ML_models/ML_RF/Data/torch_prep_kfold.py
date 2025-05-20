#!/usr/bin/env python

"""
torch_prep_kfold.py

This script prepares data for machine learning experiments in three modes:

1) --initial_split:
   - Merges feature data (optionally from multiple files) with reference data using a shared ID column.
   - Performs an optional "sequence-level scrambling" of labels in the TRAINING set only, controlled by `scramble_fractions`.
   - Splits into train and test sets by unique sequence ID (or stratified for classification).
   - Saves the resulting CSV files:
       * PREFIX_MODELTYPE_scrFRAC_trn_final.csv   (training)
       * PREFIXMODELTYPE_scrFRAC_tst_preprocess.csv (test)

2) --process train:
   - Loads the saved training CSV (with or without scrambling).
   - (Optionally) filters the last X% of rows per sequence (using `--keep_last_percent`).
   - Averages numeric columns in groups of `--navg` rows per sequence.
   - Computes mean/std for the features and standardizes them (label column is unaffected).
   - Repeats the K-fold split `--num_repeats` times to produce multiple sets of folds.
   - Saves each fold as:
        PREFIX_MODELTYPE_scrFRAC_trn_{repeat_idx}_{fold_idx}.csv
        PREFIX_MODELTYPE_scrFRAC_val_{repeat_idx}_{fold_idx}.csv

3) --process test:
   - Loads the saved test CSV (with or without scrambling).
   - Filters (last X%), averages, and standardizes using the training stats (same as above).
   - Saves the processed test file as:
        PREFIX_MODELTYPE_scrFRAC_tst_final.csv
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
# Logging Configuration
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# Argument Parsing
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for data preparation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Data preparation script with three modes:\n"
            "1) --initial_split : Merge data and do train/test split (with optional sequence-level scrambling).\n"
            "2) --process train : Process the training set (filter, average, standardize, repeated K-fold).\n"
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

    # Feature data
    parser.add_argument("--filenames", nargs="+",
                        help="CSV files containing features (used if --initial_split).")
    parser.add_argument("--feature_id_col", type=str, default="sequence",
                        help="Name of the ID column in the feature CSV(s).")
    parser.add_argument("--usecols", type=str, nargs="+", default=None,
                        help="Optional columns to load from feature CSV(s). Must include ID col if needed.")

    # Data splits / randomization
    parser.add_argument("--keep_last_percent", type=float, default=0.0,
                        help="Keep only the last X% of rows per sequence. If <=0 or >=100, keep all.")
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
                            "List of scramble fractions. We only use the first fraction. "
                            "Sequence-level scrambling means that fraction of sequences in the training set "
                            "will have their label overwritten by another sequence's label."
                        ))

    return parser.parse_args()

###############################################################################
# Data Loading (Reference + Features)
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
# Sequence-Level Scrambling
###############################################################################
def scramble_sequences(df: pd.DataFrame, id_col: str, label_col: str, frac: float, seed: int) -> pd.DataFrame:
    """
    Reassign the label for a fraction of sequences. For each scrambled sequence, pick a different
    random sequence's label and overwrite the entire sequence's label with it.

    If frac=0.3, then ~30% of the unique sequences will have their label replaced by the label
    from a different random sequence. All rows for that sequence are overwritten.

    If there's only 1 unique sequence or fraction is too small to scramble at least one,
    no scrambling occurs.

    :param df:        The DataFrame with columns [id_col, label_col].
    :param id_col:    The sequence/ID column name.
    :param label_col: The label column name.
    :param frac:      Fraction of sequences to scramble (0..1).
    :param seed:      Random seed.
    :return:          A copy of df with entire-sequence label scrambling for fraction of sequences.
    """
    if frac <= 0.0:
        return df  # no scrambling

    df_out = df.copy()
    unique_seqs = df_out[id_col].unique()
    n_seq = len(unique_seqs)

    # Number of sequences to scramble
    n_to_scramble = int(n_seq * frac)
    if n_to_scramble < 1:
        # fraction too small => do nothing
        return df_out

    rng = np.random.default_rng(seed)
    scramble_seqs = rng.choice(unique_seqs, size=n_to_scramble, replace=False)

    # For each sequence to scramble, pick a different random "source" sequence and copy its label
    for seq in scramble_seqs:
        possible_sources = [s for s in unique_seqs if s != seq]
        if not possible_sources:
            continue
        src_seq = rng.choice(possible_sources)
        # Take the label of src_seq (assuming src_seq is consistent)
        src_label = df_out.loc[df_out[id_col] == src_seq, label_col].iloc[0]
        # Overwrite the entire scrambled sequence with this new label
        df_out.loc[df_out[id_col] == seq, label_col] = src_label

    return df_out

###############################################################################
# Filtering, Averaging, and Standardization
###############################################################################
def keep_last_n_percent(df: pd.DataFrame, seq_col: str, keep_percent: float) -> pd.DataFrame:
    """
    Retain only the last keep_percent fraction of rows in each sequence group.
    If 'run' column exists, sort by it first.
    """
    if keep_percent <= 0 or keep_percent >= 100:
        return df
    if "run" in df.columns:
        df_sorted = df.sort_values([seq_col, "run"], kind="mergesort")
    else:
        df_sorted = df.copy()
    group_sizes = df_sorted.groupby(seq_col)[seq_col].transform("size")
    cumcount = df_sorted.groupby(seq_col).cumcount()
    n_keep = (group_sizes * (keep_percent / 100.0)).astype(int)
    n_keep = n_keep.mask(n_keep < 1, 1)  # ensure at least 1 row if fraction>0
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
    Shuffle the sequence's rows, chunk into size navg, and average numeric features.
    Label = label from the first row of each chunk.
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
    Apply the above averaging function per sequence group, then concat.
    """
    all_chunks = []
    for seq, group in df.groupby(id_col):
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
    Compute mean/std for numeric columns (excluding ID and label).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if id_col in numeric_cols:
        numeric_cols.remove(id_col)
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    means = [(c, df[c].mean()) for c in numeric_cols]
    stds  = [(c, df[c].std())  for c in numeric_cols]
    return (
        pd.DataFrame(means, columns=["colname", "mean"]),
        pd.DataFrame(stds, columns=["colname", "std"])
    )

def save_mean_std(mean_df: pd.DataFrame, std_df: pd.DataFrame, filename: str) -> None:
    merged = pd.merge(mean_df, std_df, on="colname")
    merged.to_csv(filename, index=False)

def load_mean_std(filename: str) -> pd.DataFrame:
    if not os.path.isfile(filename):
        logging.error(f"Mean/Std file not found: {filename}")
        sys.exit(1)
    df_stats = pd.read_csv(filename)
    needed_cols = {"colname", "mean", "std"}
    if not needed_cols.issubset(df_stats.columns):
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
    df_std = df.copy()
    means = dict(zip(stats_df["colname"], stats_df["mean"]))
    stds  = dict(zip(stats_df["colname"], stats_df["std"]))
    for col in df_std.columns:
        if col in [id_col, label_col]:
            continue
        if col in means and col in stds:
            mu  = means[col]
            sigma = stds[col]
            if sigma == 0 or np.isnan(sigma):
                df_std[col] = df_std[col] - mu
            else:
                df_std[col] = (df_std[col] - mu) / sigma
    return df_std

###############################################################################
# Main Function
###############################################################################
def main():
    args = parse_arguments()

    # We'll only use the first fraction from scramble_fractions
    scr_frac = args.scramble_fractions[0]
    frac_str = f"{scr_frac:.2f}".replace(".", "p")

    ###########################################################################
    # 1) INITIAL SPLIT
    ###########################################################################
    if args.initial_split:
        # Check inputs
        if not (args.reference_file and args.filenames):
            logging.error("Must provide --reference_file and --filenames when using --initial_split.")
            sys.exit(1)

        # Load reference data => ID + label
        df_ref = load_reference_data(args.reference_file, args.ref_id_col, args.ref_label_col)
        logging.info(f"Loaded reference data: shape={df_ref.shape}")

        # Load feature data (possibly multiple files)
        df_feat_list = []
        for f in args.filenames:
            df_tmp = load_feature_data(f, args.feature_id_col, args.usecols)
            df_feat_list.append(df_tmp)
        df_all_features = pd.concat(df_feat_list, ignore_index=True)
        logging.info(f"Combined feature data => shape={df_all_features.shape}")

        # Merge on ID col
        if args.feature_id_col != args.ref_id_col:
            df_all_features.rename(columns={args.feature_id_col: args.ref_id_col}, inplace=True)
        df_merged = pd.merge(df_all_features, df_ref, on=args.ref_id_col, how="inner")
        logging.info(f"After merging => shape={df_merged.shape}")

        # Shuffle everything
        df_merged = df_merged.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

        # Split train vs test by sequence or stratified group
        if args.model_type in ["bin", "mclass"]:
            from sklearn.model_selection import StratifiedGroupKFold
            y_for_split = df_merged[args.ref_label_col]
            groups = df_merged[args.ref_id_col]
            gkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.random_state)
            train_idx, test_idx = next(gkf.split(df_merged, y_for_split, groups=groups))
            df_train = df_merged.iloc[train_idx].copy()
            df_test  = df_merged.iloc[test_idx].copy()
        else:
            from sklearn.model_selection import GroupKFold
            unique_seqs = df_merged[args.ref_id_col].unique()
            np.random.seed(args.random_state)
            np.random.shuffle(unique_seqs)
            n_train = int((1 - args.test_percentage) * len(unique_seqs))
            train_seqs = unique_seqs[:n_train]
            test_seqs  = unique_seqs[n_train:]
            df_train = df_merged[df_merged[args.ref_id_col].isin(train_seqs)].copy()
            df_test  = df_merged[df_merged[args.ref_id_col].isin(test_seqs)].copy()

        # Now do sequence-level scrambling on the TRAIN set only
        if scr_frac > 0:
            df_train = scramble_sequences(
                df_train,
                id_col=args.ref_id_col,
                label_col=args.ref_label_col,
                frac=scr_frac,
                seed=args.random_state
            )

        # Drop 'run' column if present
        if "run" in df_train.columns:
            df_train.drop(columns=["run"], inplace=True, errors="ignore")
        if "run" in df_test.columns:
            df_test.drop(columns=["run"], inplace=True, errors="ignore")

        # Save
        train_file = f"{args.prefix}_{args.model_type}_scr{frac_str}_trn_final.csv"
        test_file  = f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_preprocess.csv"
        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)
        logging.info(f"Scr={scr_frac}: train => {train_file}, shape={df_train.shape}")
        logging.info(f"Scr={scr_frac}: test  => {test_file}, shape={df_test.shape}")
        logging.info("Initial split completed. Exiting.")
        sys.exit(0)

    ###########################################################################
    # 2) PROCESS MODE: "train" or "test"
    ###########################################################################
    if args.process is None:
        logging.error("Must specify either --initial_split or --process [train/test].")
        sys.exit(1)

    if args.process == "train":
        # We load the previously saved training CSV
        train_file = f"{args.prefix}_{args.model_type}_scr{frac_str}_trn_final.csv"
        if not os.path.isfile(train_file):
            logging.error(f"Training file not found => {train_file}")
            sys.exit(1)

        df_train = pd.read_csv(train_file)
        logging.info(f"Loaded training CSV => {train_file}, shape={df_train.shape}")
        # Drop 'run' if it exists
        if "run" in df_train.columns:
            df_train.drop(columns=["run"], inplace=True, errors="ignore")

        # Keep only the last X% if requested
        df_train = keep_last_n_percent(df_train, args.ref_id_col, args.keep_last_percent)

        # Average numeric features in chunks of size --navg
        df_train_avg = average_features_for_mutants(
            df_train,
            navg=args.navg,
            id_col=args.ref_id_col,
            label_col=args.ref_label_col,
            random_state=args.random_state
        )
        logging.info(f"After averaging => shape={df_train_avg.shape}")

        # Compute and save mean/std
        mean_df, std_df = compute_mean_std(df_train_avg, args.model_type, args.ref_id_col, args.ref_label_col)
        stats_file = f"{args.prefix}_{args.model_type}_train_stats.csv"
        save_mean_std(mean_df, std_df, stats_file)
        logging.info(f"Saved training mean/std => {stats_file}")

        # Apply standardization
        df_train_std = apply_standardization(
            df_train_avg,
            load_mean_std(stats_file),
            args.model_type,
            args.ref_id_col,
            args.ref_label_col
        )
        # Shuffle once more
        df_train_std = df_train_std.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

        # Repeated K-fold
        if args.model_type in ["bin", "mclass"]:
            from sklearn.model_selection import StratifiedGroupKFold
            y_for_split = df_train_std[args.ref_label_col]
            groups      = df_train_std[args.ref_id_col]
        else:
            from sklearn.model_selection import GroupKFold
            groups = df_train_std[args.ref_id_col]

        for repeat_idx in range(args.num_repeats):
            # (Optional) offset seed if you want different splits each repeat
            repeat_seed = args.random_state + 100 * repeat_idx

            if args.model_type in ["bin", "mclass"]:
                kf = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=repeat_seed)
                split_iter = kf.split(df_train_std, y_for_split, groups)
            else:
                kf = GroupKFold(n_splits=args.kfold)
                split_iter = kf.split(df_train_std, groups=groups)

            fold_counter = 0
            for trn_idx, val_idx in split_iter:
                df_fold_trn = df_train_std.iloc[trn_idx].copy()
                df_fold_val = df_train_std.iloc[val_idx].copy()

                col_order = [args.ref_id_col] + [c for c in df_fold_trn.columns if c != args.ref_id_col]
                df_fold_trn = df_fold_trn[col_order]
                df_fold_val = df_fold_val[col_order]

                fold_train_csv = f"{args.prefix}_{args.model_type}_scr{frac_str}_trn_{repeat_idx}_{fold_counter}.csv"
                fold_val_csv   = f"{args.prefix}_{args.model_type}_scr{frac_str}_val_{repeat_idx}_{fold_counter}.csv"
                df_fold_trn.to_csv(fold_train_csv, index=False)
                df_fold_val.to_csv(fold_val_csv, index=False)
                logging.info(
                    f"Repeat={repeat_idx}, Fold={fold_counter}: Train shape={df_fold_trn.shape}, Val shape={df_fold_val.shape}"
                )
                fold_counter += 1

        logging.info(f"Done processing training data with repeated K-fold => num_repeats={args.num_repeats}, kfold={args.kfold}")
        sys.exit(0)

    elif args.process == "test":
        # We load the previously saved test CSV
        test_file = f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_preprocess.csv"
        if not os.path.isfile(test_file):
            logging.error(f"Test file not found => {test_file}")
            sys.exit(1)

        df_test = pd.read_csv(test_file)
        logging.info(f"Loaded test CSV => {test_file}, shape={df_test.shape}")

        if "run" in df_test.columns:
            df_test.drop(columns=["run"], inplace=True, errors="ignore")

        df_test = keep_last_n_percent(df_test, args.ref_id_col, args.keep_last_percent)
        df_test_avg = average_features_for_mutants(
            df_test,
            navg=args.navg,
            id_col=args.ref_id_col,
            label_col=args.ref_label_col,
            random_state=args.random_state
        )
        logging.info(f"After averaging => shape={df_test_avg.shape}")

        # Load the training stats and apply standardization
        stats_file = f"{args.prefix}_{args.model_type}_train_stats.csv"
        if not os.path.isfile(stats_file):
            logging.error(f"Train stats not found => {stats_file}")
            sys.exit(1)

        df_test_std = apply_standardization(
            df_test_avg,
            load_mean_std(stats_file),
            args.model_type,
            args.ref_id_col,
            args.ref_label_col
        )
        df_test_std = df_test_std.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

        final_test_csv = f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_final.csv"
        df_test_std.to_csv(final_test_csv, index=False)
        logging.info(f"Final test CSV saved: {final_test_csv}, shape={df_test_std.shape}")
        sys.exit(0)


if __name__ == "__main__":
    main()
