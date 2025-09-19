"""
run_model_rf.py  –  Random-Forest train / evaluate
compatible with torch_prep_kfold_noise.py  (scr- and noi-tags)
"""

import argparse, csv, logging, os, sys
from typing import Tuple, List, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (mean_squared_error, r2_score,
                             matthews_corrcoef, accuracy_score)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# ───────────────────────────── logging ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s"
)

# ───────────────────── helper: nice float formatter ─────────────────
fmt_float = lambda x: f"{x:.4f}" if np.isfinite(x) else "NaN"
majority  = lambda v: int(pd.Series(v).mode()[0])

# ───────────────────── helper: scr-tag formatter  ───────────────────
def scr_tag(frac: float) -> str:
    """0.0 → scr0p0   0.25 → scr0p25   1.0 → scr1p0"""
    
    return "scr" + str(frac).replace(".", "p")

# ───────────────────── argument parsing  ────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Random-Forest train / eval   (noise experiments)"
    )
    p.add_argument("--mode", type=int, choices=[0,1], default=0,
                   help="0=train  1=evaluate")
    p.add_argument("--model_type", choices=["reg","bin","mclass"],
                   default="reg")
    p.add_argument("--data_scale", choices=["log","nonlog"],
                   default="log")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--num_repeats", type=int, default=5)
    p.add_argument("--model_dir", default="Model")
    p.add_argument("--data_dir",  default="Data")
    p.add_argument("--output_file", default="predictions")
    p.add_argument("--ref_id_col",    default="sequence")
    p.add_argument("--ref_label_col", default="label")
    # RF hyper-params
    p.add_argument("--n_estimators",      type=int, default=100)
    p.add_argument("--max_depth",         default="None")
    p.add_argument("--max_features",      default="auto")
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf",  type=int, default=1)
    p.add_argument("--random_state",      type=int, default=42)
    # naming
    p.add_argument("--prefix", default="gbsa")
    p.add_argument("--scramble_fractions", nargs="+", type=float,
                   default=[0.0])
    p.add_argument("--noise_tags", nargs="+", default=["noi00"],
                   help="noi00  noi05  ... must match filenames from prep")
    return p.parse_args()

# ───────────────────── I/O  helpers ─────────────────────────────────
def load_xy(csv_path:str, args) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    df  = pd.read_csv(csv_path)
    X   = df.drop(columns=[args.ref_id_col, args.ref_label_col]).to_numpy(np.float32)
    y   = df[args.ref_label_col].to_numpy(np.float32)
    ids = df[args.ref_id_col].tolist()
    return X, y, ids

def save_preds(rows, fn):
    with open(fn, "w", newline="") as f:
        wr = csv.writer(f);  wr.writerow(["Label","Predicted","True"])
        wr.writerows(rows)

# ───────────────────── RF builder ────────────────────────────────────
def build_rf(args):
    md = None if args.max_depth.lower()=="none" else int(args.max_depth)
    try: mf = float(args.max_features)
    except ValueError: mf = args.max_features
    cls = RandomForestClassifier if args.model_type!="reg" else RandomForestRegressor
    return cls(n_estimators=args.n_estimators, max_depth=md,
               max_features=mf, min_samples_split=args.min_samples_split,
               min_samples_leaf=args.min_samples_leaf,
               random_state=args.random_state)

# ───────────────────── evaluation ───────────────────────────────────
def eval_model(rf, X, y, ids, args):
    """return MSE, R2, Pearson, MCC, Acc,  row-list"""
    from math import isnan
    rows   = []
    id2y_p = defaultdict(list)
    id2y_t = defaultdict(list)

    if args.model_type=="reg":
        yhat = rf.predict(X)
    elif args.model_type=="bin":
        yhat = rf.predict(X).astype(int)
    else:                              # mclass
        yhat = rf.predict_proba(X).argmax(1).astype(int)

    for uid, phat, ytrue in zip(ids, yhat, y):
        rows.append((uid, float(phat), float(ytrue)))
        id2y_p[uid].append(phat)
        id2y_t[uid].append(ytrue)

    # aggregate per-sequence
    agg_p = [np.mean(v) if args.model_type=="reg" else majority(v)
             for v in id2y_p.values()]
    agg_t = [np.mean(v) if args.model_type=="reg" else majority(v)
             for v in id2y_t.values()]

    mse=r2=pear=mcc=acc=np.nan
    if args.model_type=="reg":
        mse = mean_squared_error(agg_t, agg_p)
        if len(agg_p)>1:
            r2  = r2_score(agg_t, agg_p)
            pear= pearsonr(agg_t, agg_p)[0]
        thr   = 0.0 if args.data_scale=="log" else 1.0
        mcc   = matthews_corrcoef(np.array(agg_t)>thr, np.array(agg_p)>thr)
        acc   = accuracy_score     (np.array(agg_t)>thr, np.array(agg_p)>thr)
    else:
        mcc   = matthews_corrcoef(agg_t, agg_p)
        acc   = accuracy_score    (agg_t, agg_p)

    return mse,r2,pear,mcc,acc,rows

# ───────────────────── training loop ────────────────────────────────
def train_loop(args):
    os.makedirs(args.model_dir, exist_ok=True)
    for scr in args.scramble_fractions:
        s_tag = scr_tag(scr)
        for noi in args.noise_tags:
            logging.info(f"── TRAIN  {s_tag} | {noi} ──")
            all_rows, all_mets = [], []
            for rep in range(args.num_repeats):
                for fold in range(args.kfold):
                    tr = f"{args.prefix}_{args.model_type}_{s_tag}_{noi}_trn_{rep}_{fold}.csv"
                    va = f"{args.prefix}_{args.model_type}_{s_tag}_{noi}_val_{rep}_{fold}.csv"
                    tr,va = os.path.join(args.data_dir,tr), os.path.join(args.data_dir,va)
                    if not (os.path.isfile(tr) and os.path.isfile(va)):
                        logging.warning(f"missing {tr}")
                        continue
                    Xtr,ytr,_ = load_xy(tr,args)
                    Xva,yva,ids= load_xy(va,args)
                    if args.model_type!="reg":
                        ytr=ytr.astype(int); yva=yva.astype(int)
                    rf = build_rf(args); rf.fit(Xtr,ytr)

                    mdl = f"rf_fold_{rep}_{fold}_{args.model_type}_{s_tag}_{noi}.pkl"
                    joblib.dump(rf, os.path.join(args.model_dir, mdl))

                    mse,r2,pr,mcc,ac,rows = eval_model(rf,Xva,yva,ids,args)
                    all_rows.extend(rows);  all_mets.append(
                        dict(MSE=mse,R2=r2,Pear=pr,MCC=mcc,Acc=ac) )

                    logging.info(f"rep={rep} fold={fold}  "
                                 f"MSE={fmt_float(mse)}  R2={fmt_float(r2)}  "
                                 f"Pear={fmt_float(pr)}  MCC={fmt_float(mcc)}  Acc={fmt_float(ac)}")

            # ─── write aggregated prediction CSV & metrics ─────────────────
            if all_rows:
                save_preds(all_rows,
                   f"{args.output_file}_{args.model_type}_{s_tag}_{noi}.csv")
            if all_mets:
                df = pd.DataFrame(all_mets).mean()
                df.to_csv(f"final_metrics_{args.model_type}_{s_tag}_{noi}_trn.csv",
                          header=False)
# ───────────────────── evaluation loop ──────────────────────────────
def eval_loop(args):
    for scr in args.scramble_fractions:
        s_tag = scr_tag(scr)
        test_csv = os.path.join(args.data_dir,
                                f"{args.prefix}_{args.model_type}_{s_tag}_tst_final.csv")
        if not os.path.isfile(test_csv):
            logging.error(f"test CSV not found: {test_csv}")
            continue
        Xte, yte, ids = load_xy(test_csv,args)
        if args.model_type!="reg": yte=yte.astype(int)

        for noi in args.noise_tags:
            logging.info(f"── EVAL  {s_tag} | {noi} ──")
            rows, mets = [], []
            for rep in range(args.num_repeats):
                for fold in range(args.kfold):
                    mdl = os.path.join(args.model_dir,
                        f"rf_fold_{rep}_{fold}_{args.model_type}_{s_tag}_{noi}.pkl")
                    if not os.path.isfile(mdl): continue
                    rf  = joblib.load(mdl)
                    mse,r2,pr,mcc,ac,r = eval_model(rf,Xte,yte,ids,args)
                    rows.extend(r); mets.append(
                        dict(MSE=mse,R2=r2,Pear=pr,MCC=mcc,Acc=ac) )
            if rows:
                save_preds(rows,
                   f"{args.output_file}_test_{args.model_type}_{s_tag}_{noi}.csv")
            if mets:
                df = pd.DataFrame(mets).mean()
                df.to_csv(f"final_metrics_{args.model_type}_{s_tag}_{noi}_tst.csv",
                          header=False)

# ───────────────────── main ─────────────────────────────────────────
def main():
    A = get_args()
    os.makedirs(A.data_dir,  exist_ok=True)
    os.makedirs(A.model_dir, exist_ok=True)
    if A.mode==0: train_loop(A)
    else:         eval_loop(A)

if __name__ == "__main__":
    main()
