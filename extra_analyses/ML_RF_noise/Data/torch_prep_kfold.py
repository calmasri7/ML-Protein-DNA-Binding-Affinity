#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
torch_prep_kfold_noise.py  ──  data-set builder with *label noise* support
--------------------------------------------------------------------------

Three entry points (exactly as before)

1) --initial_split    : merge features + reference → 85/15 train/test split
                        (plus optional *sequence-level* scrambling).

2) --process train    :   • average MD frames (navg)  
                          • standardise features  
                          • inject Gaussian / flip noise **per noise-level**  
                          • build repeated K-fold CSVs

3) --process test     : preprocess the hold-out set using the
                        stats (μ,σ) computed from the *clean* training data
                        (noise is **never** added to the test-set).

---------------------------------------------------------------------------
New command-line flags
---------------------------------------------------------------------------
--noise_levels    0.0 0.05 0.10 ...
                  ▸ Regression: add N(0, σ·level) to every ΔΔG in *train*.
                    (σ = stdev of the training labels)
                  ▸ Classification (bin / multi-class):
                    randomly flip `level` fraction of *sequences*.

                  Filenames include a tag  «noi05»  (=5 % noise).

---------------------------------------------------------------------------
File naming scheme (train/val folds)
---------------------------------------------------------------------------
gbsa_<mtype>_scr0p00_noi05_trn_0_2.csv
gbsa_<mtype>_scr0p00_noi05_val_0_2.csv
     ^            ^     ^      ^ ^
     |            |     |      | └─ fold index
     |            |     |      └──── repeat index
     |            |     └─────────── noise tag  (noi00, noi05, …)
     |            └───────────────── scramble fraction (unchanged)
     └────────────────────────────── model-type  (reg / bin / mclass)
"""

###############################################################################
# Std-lib / 3rd-party imports
###############################################################################
import argparse, logging, os, sys
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s"
)

###############################################################################
# -------------------------  Argument parsing  ------------------------------ #
###############################################################################
def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prepare ML CSVs with optional sequence scrambling AND label noise."
    )

    # ── files / columns ────────────────────────────────────────────────────
    p.add_argument("--reference_file",     type=str,               help="Reference CSV (ID + label)")
    p.add_argument("--ref_id_col",         type=str, default="sequence")
    p.add_argument("--ref_label_col",      type=str, default="label")

    p.add_argument("--filenames",          nargs="+",               help="Feature CSV(s)")
    p.add_argument("--feature_id_col",     type=str, default="sequence")
    p.add_argument("--usecols",            nargs="+", default=None)

    # ── data-handling options ──────────────────────────────────────────────
    p.add_argument("--keep_last_percent",  type=float, default=0.0,
                   help="Keep last %% of frames per sequence before averaging")
    p.add_argument("--navg",               type=int,   default=50,
                   help="Frames to average into one row")
    p.add_argument("--test_percentage",    type=float, default=0.15)

    # ── ML / CV parameters ─────────────────────────────────────────────────
    p.add_argument("--model_type",         choices=["reg","bin","mclass"], required=True)
    p.add_argument("--kfold",              type=int, default=5)
    p.add_argument("--num_repeats",        type=int, default=1)
    p.add_argument("--random_state",       type=int, default=42)

    # ── label-scrambling & noise ───────────────────────────────────────────
    p.add_argument("--scramble_fractions", type=float, nargs="+", default=[0.0],
                   help="Fraction of *train* sequences whose labels are replaced by another's")
    p.add_argument("--noise_levels",       type=float, nargs="+", default=[0.0],
                   help="Label-noise levels  (0.05 = 5 %)")

    # ── modes & misc ───────────────────────────────────────────────────────
    p.add_argument("--prefix",             type=str, default="gbsa")
    p.add_argument("--initial_split",      action="store_true")
    p.add_argument("--process",            choices=["train","test"])

    return p.parse_args()

###############################################################################
# -----------------------  I/O utilities (unchanged)  ----------------------- #
###############################################################################
def load_reference_data(fname:str, id_col:str, lab_col:str) -> pd.DataFrame:
    df = pd.read_csv(fname)
    if id_col not in df or lab_col not in df:
        logging.error(f"Reference CSV missing {id_col}/{lab_col}")
        sys.exit(1)
    return df[[id_col, lab_col]].copy()

def load_feature_data(fname:str, id_col:str, usecols:Optional[List[str]]) -> pd.DataFrame:
    df = pd.read_csv(fname, usecols=usecols) if usecols else pd.read_csv(fname)
    if id_col not in df:
        logging.error(f"Feature CSV {fname} missing {id_col}")
        sys.exit(1)
    return df

###############################################################################
# -------------------  Helper #1  – sequence-scrambling  -------------------- #
###############################################################################
def scramble_sequences(df:pd.DataFrame, id_col:str, lab_col:str,
                       frac:float, seed:int) -> pd.DataFrame:
    """Copy the label of a *different* random sequence into `frac` of sequences."""
    if frac <= 0.0:
        return df
    out      = df.copy()
    seqs     = out[id_col].unique()
    n_scramb = max(1, int(len(seqs)*frac))
    rng      = np.random.default_rng(seed)
    target   = rng.choice(seqs, size=n_scramb, replace=False)
    for tgt in target:
        donor = rng.choice([s for s in seqs if s != tgt])
        new_y = out.loc[out[id_col]==donor, lab_col].iloc[0]
        out.loc[out[id_col]==tgt, lab_col] = new_y
    return out

###############################################################################
# -------------------  Helper #2  – keep-last-percent  ---------------------- #
###############################################################################
def keep_last_n_percent(df:pd.DataFrame, seq_col:str, pct:float) -> pd.DataFrame:
    if pct<=0 or pct>=100:  # keep all
        return df
    sorter = ["run"] if "run" in df.columns else []
    df2    = df.sort_values([seq_col]+sorter, kind="mergesort")
    grp_sz = df2.groupby(seq_col)[seq_col].transform("size")
    cut    = (grp_sz * (pct/100)).astype(int).clip(lower=1)
    mask   = df2.groupby(seq_col).cumcount() >= (grp_sz - cut)
    return df2[mask].reset_index(drop=True)

###############################################################################
# -------------------  Helper #3  – frame-averaging  ------------------------ #
###############################################################################
def average_features_for_sequence(df, navg, id_col, lab_col, seed):
    rs    = np.random.default_rng(seed)
    df_sh = df.sample(frac=1.0, random_state=rs.integers(0,1e6))
    num   = df_sh.select_dtypes(np.number).columns.difference([lab_col])
    chunks= len(df_sh)//navg
    out   = []
    for i in range(chunks):
        ch   = df_sh.iloc[i*navg:(i+1)*navg]
        row  = {col: ch[col].mean() for col in num}
        row[lab_col] = ch[lab_col].iloc[0]
        row[id_col]  = ch[id_col].iloc[0]
        out.append(row)
    return pd.DataFrame(out)

def average_features_for_mutants(df, navg, id_col, lab_col, seed):
    pieces=[]
    for _,grp in df.groupby(id_col):
        pieces.append(average_features_for_sequence(grp, navg, id_col, lab_col, seed))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

###############################################################################
# --------------------  Helper #4  – standardisation  ----------------------- #
###############################################################################
def compute_mean_std(df, id_col, lab_col):
    num = df.select_dtypes(np.number).columns.difference([lab_col])
    return df[num].mean().to_frame("mean"), df[num].std().to_frame("std")

def apply_standardisation(df, stats, id_col, lab_col):
    num = df.select_dtypes(np.number).columns.difference([lab_col])
    mu  = stats["mean"].squeeze()
    sd  = stats["std"].squeeze().replace(0, 1)
    df[num] = (df[num]-mu)/sd
    return df

###############################################################################
# --------------------  Helper #5  – label-noise  --------------------------- #
###############################################################################
def add_label_noise(df:pd.DataFrame, level:float, model_type:str,
                    id_col:str, lab_col:str, seed:int)->pd.DataFrame:
    """Inject noise *only in training*:
       • regression :  y ← y +  N(0, σ·level)
       • bin/mclass :  flip `level` fraction of sequence labels
    """
    if level<=0:
        return df
    rng = np.random.default_rng(seed)
    out = df.copy()

    if model_type=="reg":
        sigma = out[lab_col].std(ddof=0)
        out[lab_col] += rng.normal(0, sigma*level, size=len(out))
    else:
        seqs = out[id_col].unique()
        nflp = int(len(seqs)*level)
        flip = rng.choice(seqs, size=max(1,nflp), replace=False)
        for s in flip:
            grp_idx = out[id_col]==s
            if model_type=="bin":
                out.loc[grp_idx, lab_col] = 1 - out.loc[grp_idx, lab_col]
            else:                   # multi-class – random label ≠ old
                labels      = sorted(out[lab_col].unique())
                old         = out.loc[grp_idx, lab_col].iloc[0]
                new_label   = rng.choice([l for l in labels if l!=old])
                out.loc[grp_idx, lab_col] = new_label
    return out

###############################################################################
# -----------------------------  main  -------------------------------------- #
###############################################################################
def main():
    A   = parse_arguments()
    rng = np.random.default_rng(A.random_state)

    scr_frac = A.scramble_fractions[0]
    scr_tag  = f"scr{str(scr_frac).replace('.','p'):>}"

    # ───────────────────────── 1) initial split ────────────────────────────
    if A.initial_split:
        if not (A.reference_file and A.filenames):
            logging.error("Need --reference_file and --filenames with --initial_split")
            sys.exit(1)

        df_ref = load_reference_data(A.reference_file, A.ref_id_col, A.ref_label_col)

        feats  = [load_feature_data(f, A.feature_id_col, A.usecols) for f in A.filenames]
        df_feat= pd.concat(feats, ignore_index=True)

        if A.feature_id_col != A.ref_id_col:
            df_feat = df_feat.rename(columns={A.feature_id_col:A.ref_id_col})

        df = pd.merge(df_feat, df_ref, on=A.ref_id_col, how="inner")\
               .sample(frac=1.0, random_state=A.random_state).reset_index(drop=True)

        # ----- split by unique sequences -----
        seqs = df[A.ref_id_col].unique()
        rng.shuffle(seqs)
        n_tr = int(len(seqs)*(1-A.test_percentage))
        tr_seqs, te_seqs = seqs[:n_tr], seqs[n_tr:]

        df_tr = df[df[A.ref_id_col].isin(tr_seqs)].copy()
        df_te = df[df[A.ref_id_col].isin(te_seqs)].copy()

        # optional scramble on *train*
        df_tr = scramble_sequences(df_tr, A.ref_id_col, A.ref_label_col,
                                   scr_frac, A.random_state)

        df_tr.to_csv(f"{A.prefix}_{A.model_type}_{scr_tag}_trn_final.csv", index=False)
        df_te.to_csv(f"{A.prefix}_{A.model_type}_{scr_tag}_tst_preprocess.csv", index=False)
        logging.info("Initial split done.")
        sys.exit(0)

    # ───────────────────────── 2) process=train ────────────────────────────
    if A.process=="train":
        train_csv = f"{A.prefix}_{A.model_type}_{scr_tag}_trn_final.csv"
        if not os.path.exists(train_csv):
            logging.error(f"Missing {train_csv}; run --initial_split first.")
            sys.exit(1)

        df   = pd.read_csv(train_csv)
        df   = keep_last_n_percent(df, A.ref_id_col, A.keep_last_percent)
        df   = average_features_for_mutants(df, A.navg,
                                            A.ref_id_col, A.ref_label_col, A.random_state)

        mu, sd = compute_mean_std(df, A.ref_id_col, A.ref_label_col)
        stats  = pd.concat([mu, sd["std"]], axis=1)
        stats.to_csv(f"{A.prefix}_{A.model_type}_train_stats.csv")

        df_std = apply_standardisation(df, stats, A.ref_id_col, A.ref_label_col)

        # -------- iterate over noise levels --------------------------------
        for level in sorted(set(A.noise_levels)):
            noi_tag = f"noi{int(level*100):02d}"
            df_noi  = add_label_noise(df_std, level, A.model_type,
                                      A.ref_id_col, A.ref_label_col,
                                      A.random_state+int(level*1000))
            logging.info(f"[{noi_tag}] after noise σ/flip={level:.2f} → shape={df_noi.shape}")

            # choose CV splitter
            if A.model_type in ["bin","mclass"]:
                splitter = StratifiedGroupKFold(n_splits=A.kfold,
                                                shuffle=True,
                                                random_state=A.random_state)
                split_iter = splitter.split(df_noi,
                                            df_noi[A.ref_label_col],
                                            groups=df_noi[A.ref_id_col])
            else:
                splitter = GroupKFold(n_splits=A.kfold)
                split_iter = splitter.split(df_noi,
                                            groups=df_noi[A.ref_id_col])

            # repeated K-fold
            for rep in range(A.num_repeats):
                if A.model_type in ["bin","mclass"]:
                    splitter = StratifiedGroupKFold(n_splits=A.kfold,
                                                    shuffle=True,
                                                    random_state=A.random_state+rep)
                    split_iter = splitter.split(df_noi,
                                                df_noi[A.ref_label_col],
                                                groups=df_noi[A.ref_id_col])
                for fold, (idx_tr, idx_va) in enumerate(split_iter):
                    df_tr = df_noi.iloc[idx_tr].copy()
                    df_va = df_noi.iloc[idx_va].copy()

                    fn_tr = (f"{A.prefix}_{A.model_type}_{scr_tag}_{noi_tag}"
                             f"_trn_{rep}_{fold}.csv")
                    fn_va = (f"{A.prefix}_{A.model_type}_{scr_tag}_{noi_tag}"
                             f"_val_{rep}_{fold}.csv")
                    df_tr.to_csv(fn_tr, index=False)
                    df_va.to_csv(fn_va, index=False)
                    logging.info(f"[{noi_tag}] rep={rep}, fold={fold} →"
                                 f" {df_tr.shape} / {df_va.shape}")

    # ───────────────────────── 3) process=test ────────────────────────────
    elif A.process=="test":
        test_csv = f"{A.prefix}_{A.model_type}_{scr_tag}_tst_preprocess.csv"
        if not os.path.exists(test_csv):
            logging.error("Run --initial_split first.")
            sys.exit(1)

        df_te = pd.read_csv(test_csv)
        df_te = keep_last_n_percent(df_te, A.ref_id_col, A.keep_last_percent)
        df_te = average_features_for_mutants(df_te, A.navg,
                                             A.ref_id_col, A.ref_label_col, A.random_state)

        stats = pd.read_csv(
            f"{A.prefix}_{A.model_type}_train_stats.csv",
            index_col=0          
        )
        df_te = apply_standardisation(df_te, stats, A.ref_id_col, A.ref_label_col)
        df_te.to_csv(f"{A.prefix}_{A.model_type}_{scr_tag}_tst_final.csv",
                     index=False)
        logging.info("Test-set preprocessing done.")

if __name__ == "__main__":
    main()
