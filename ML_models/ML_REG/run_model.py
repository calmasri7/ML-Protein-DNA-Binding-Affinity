#!/usr/bin/env python
"""
run_model_linear.py  

Train or evaluate a linear model (ordinary least-squares regression) or,
more generally, a single-layer network when the user sets `--hidden_layers 0`.
If `--hidden_layers > 0`, the script behaves as a normal MLP but keeps
the option open for strictly linear models.

The workflow is identical to the previously annotated NN script:

* repeated K-fold cross-validation  
* optional scramble fractions to test information leakage  
* support for regression, binary- and multi-class classification  
* optional early stopping (validation loss)  
* automatic aggregation from row-level to ID-level predictions  
* CSV reports for row predictions and averaged metrics  

Only explanatory comments have been added – inputs, outputs, and all
functional behavior remain exactly the same.
"""

###############################################################################
# Imports
###############################################################################
import argparse, sys, os, csv, logging
from typing import Tuple, List, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch as T

from sklearn.metrics import (matthews_corrcoef, accuracy_score, r2_score)
from scipy.stats import pearsonr

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.DEBUG,                     # DEBUG → very verbose; change if needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# Helper utilities
###############################################################################
def fmt_float(x: float) -> str:
    """
    Format *x* with four decimals unless NaN – return string 'NaN' in that case.
    """
    return f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "NaN"


def majority_vote(values: List[int]) -> int:
    """
    Return the most common integer in *values*.
    `pandas.Series.mode()` returns *all* modes; `[0]` picks the first in a tie.
    """
    return int(pd.Series(values).mode()[0])

###############################################################################
# Command-line interface
###############################################################################
def parse_args():
    """
    Build and parse CLI arguments.  Any hyper-parameter or file path can be
    overridden from the shell instead of editing the script.
    """
    parser = argparse.ArgumentParser(
        description="Linear (or simple NN) model – training / evaluation"
    )

    # ----------------------- core flags ---------------------------------- #
    parser.add_argument("--mode", type=int, default=0,
                        help="0 = training, 1 = evaluation.")
    parser.add_argument("--model_type", type=str, default="reg",
                        choices=["reg", "bin", "mclass"],
                        help="'reg' = regression, "
                             "'bin' = binary cls, "
                             "'mclass' = multi-class cls.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Only used for --model_type mclass.")
    parser.add_argument("--data_scale", type=str, default="log",
                        choices=["log", "nonlog"],
                        help="Determines threshold when binarizing regression outputs.")

    # ----------------------- cross-validation ---------------------------- #
    parser.add_argument("--kfold", type=int, default=5,
                        help="#folds for K-fold CV.")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="#times to repeat the K-fold split.")
    parser.add_argument("--scramble_fractions", type=float, nargs="+",
                        default=[0.0],
                        help="Fractions used during data-prep; "
                             "filenames contain e.g. 'scr0p25'.")

    # These hyper-parameters remain from the generic NN implementation.
    # For a purely linear model you can ignore most of them (e.g., dropout),
    # but they do no harm if hidden_layers == 0.
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--lrn_rate", type=float, default=1e-4)
    parser.add_argument("--wt_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_input_output", type=float, default=0.1)
    parser.add_argument("--dropout_hidden", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=44)
    parser.add_argument("--hidden_layers", type=int, default=3,
                        help="Set 0 for *strictly linear* model.")
    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=50)

    # thresholds to derive ± class labels from regression predictions
    parser.add_argument("--regression_threshold_log", type=float, default=0.0)
    parser.add_argument("--regression_threshold_nonlog", type=float, default=1.0)

    # ----------------------- paths --------------------------------------- #
    parser.add_argument("--model_dir", type=str, default="Model/")
    parser.add_argument("--data_dir",  type=str, default="Data/")
    parser.add_argument("--output_file", type=str, default="predictions")
    parser.add_argument("--prefix", type=str, default="gbsa")

    # ----------------------- CSV column names --------------------------- #
    parser.add_argument("--ref_id_col",    type=str, default="sequence")
    parser.add_argument("--ref_label_col", type=str, default="label")

    return parser.parse_args()

###############################################################################
# Data loading
###############################################################################
def load_csv_data(csv_file: str, args) -> Tuple[T.Tensor, T.Tensor, List[Any]]:
    """
    Convert CSV → tensors.

    Expected columns:
        * args.ref_id_col    – unique ID (e.g. sequence) – *kept as list*
        * args.ref_label_col – target / label
        * everything else    – numeric features
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Data file missing: {csv_file}")

    df = pd.read_csv(csv_file, header=0)
    id_col, lbl_col = args.ref_id_col, args.ref_label_col

    if id_col not in df.columns or lbl_col not in df.columns:
        raise ValueError(f"CSV {csv_file} must contain '{id_col}' & '{lbl_col}'.")

    feat_cols = [c for c in df.columns if c not in [id_col, lbl_col]]
    ids      = df[id_col].tolist()
    X        = T.tensor(df[feat_cols].values.astype(np.float32))
    y        = T.tensor(df[lbl_col].values.astype(np.float32)).unsqueeze(1)
    return X, y, ids

###############################################################################
# Model definition
###############################################################################
class Net(T.nn.Module):
    """
    *hidden_layers = 0* → single Linear layer → purely linear regression  
    *hidden_layers ≥ 1* → classic MLP with ReLU + dropout.
    """
    def __init__(self, input_dim: int, args):
        super().__init__()
        out_dim = args.num_classes if args.model_type == "mclass" else 1
        self.args = args

        # ---------- strictly linear case -------------------------------- #
        if args.hidden_layers == 0:
            self.layers = T.nn.ModuleList([T.nn.Linear(input_dim, out_dim)])
            self.act = self.dropout_io = self.dropout_hidden = None

        # ---------- standard MLP --------------------------------------- #
        else:
            layers = [T.nn.Linear(input_dim, args.hidden_size)]
            for _ in range(args.hidden_layers - 1):
                layers.append(T.nn.Linear(args.hidden_size, args.hidden_size))
            layers.append(T.nn.Linear(args.hidden_size, out_dim))
            self.layers = T.nn.ModuleList(layers)

            self.act = T.nn.ReLU()
            self.dropout_io  = T.nn.Dropout(args.dropout_input_output)
            self.dropout_hidden = T.nn.Dropout(args.dropout_hidden)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for all Linear layers."""
        for layer in self.layers:
            T.nn.init.xavier_uniform_(layer.weight)
            T.nn.init.zeros_(layer.bias)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Forward propagation – two branches depending on hidden_layers.
        """
        # -------- strictly linear ------------------------------------- #
        if self.args.hidden_layers == 0:
            return self.layers[0](x)

        # -------- MLP ------------------------------------------------- #
        x = self.dropout_io(self.act(self.layers[0](x)))
        for layer in self.layers[1:-1]:
            x = self.dropout_hidden(self.act(layer(x)))
        return self.layers[-1](x)      # logits or regression value

###############################################################################
# Loss function selector
###############################################################################
def get_loss_function(args):
    if args.model_type == "reg":
        return T.nn.MSELoss()
    elif args.model_type == "bin":
        return T.nn.BCEWithLogitsLoss()
    elif args.model_type == "mclass":
        return T.nn.CrossEntropyLoss()
    raise ValueError(f"Unknown model_type '{args.model_type}'")

###############################################################################
# Early stopping helper
###############################################################################
class EarlyStopper:
    """
    Stop training when validation loss hasn't improved after *patience* epochs.
    Keeps a copy of the best model state_dict.
    """
    def __init__(self, patience=20, mode="min"):
        self.patience = patience
        self.mode     = mode
        self.best_value = None
        self.counter    = 0
        self.should_stop = False
        self.best_model_state = None

    def check(self, val, state_dict):
        # first observation
        if self.best_value is None:
            self.best_value, self.best_model_state = val, state_dict
            return

        improved = (val < self.best_value) if self.mode == "min" else (val > self.best_value)
        if improved:
            self.best_value, self.best_model_state = val, state_dict
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

###############################################################################
# Training loop
###############################################################################
def train_model(net, X_trn, y_trn, X_val, y_val, args):
    """
    Full-batch gradient descent with optional early stopping.
    """
    crit = get_loss_function(args)
    optim = T.optim.Adam(net.parameters(), lr=args.lrn_rate, weight_decay=args.wt_decay)

    if args.model_type == "mclass":
        y_trn = y_trn.view(-1).long()
        y_val = y_val.view(-1).long()

    early = EarlyStopper(args.patience, mode="min") if args.use_early_stopping else None
    log_every = max(1, args.max_epochs // 10)

    net.train()
    for ep in range(args.max_epochs):
        optim.zero_grad()
        loss = crit(net(X_trn), y_trn)
        loss.backward()
        optim.step()

        if (ep + 1) % log_every == 0:
            logging.info(f"[Epoch {ep+1}/{args.max_epochs}] loss={loss.item():.4f}")

        if early:
            net.eval()
            with T.no_grad():
                val_loss = crit(net(X_val), y_val).item()
            net.train()

            early.check(val_loss, net.state_dict())
            if early.should_stop:
                logging.info(f"Early stop at epoch {ep+1} "
                             f"(best val-loss={early.best_value:.4f})")
                net.load_state_dict(early.best_model_state)
                break

###############################################################################
# Evaluation routine
###############################################################################
def evaluate_model(net, X, y, ids, args):
    """
    Predict on *X* and compute task-appropriate metrics.
    Row-level predictions are aggregated to unique IDs.
    Returns:
        (mse, r2, pearson, mcc, accuracy, row_level_pred_list)
    """
    net.eval()
    with T.no_grad():
        out = net(X)

    # ----------------------------------------------------------------- #
    # MULTI-CLASS CLASSIFICATION
    # ----------------------------------------------------------------- #
    if args.model_type == "mclass":
        logits = out.cpu().numpy()
        tgt    = y.view(-1).cpu().numpy()
        row_pred = np.argmax(logits, axis=1)

        agg = defaultdict(lambda: {"logits": [], "tgt": []})
        rows = []
        for i, uid in enumerate(ids):
            agg[uid]["logits"].append(logits[i])
            agg[uid]["tgt"].append(int(tgt[i]))
            rows.append((uid, row_pred[i], tgt[i]))

        preds, tgts = [], []
        for d in agg.values():
            preds.append(int(np.argmax(np.sum(d["logits"], axis=0))))
            tgts .append(majority_vote(d["tgt"]))

        mcc = acc = float('nan')
        if len(set(tgts)) > 1:
            mcc = matthews_corrcoef(tgts, preds)
            acc = accuracy_score(tgts, preds)
        return (np.nan, np.nan, np.nan, mcc, acc, rows)

    # ----------------------------------------------------------------- #
    # BINARY CLASSIFICATION
    # ----------------------------------------------------------------- #
    if args.model_type == "bin":
        logits = out.cpu().numpy().flatten()
        tgt    = y.cpu().numpy().flatten()
        row_pred = (logits > 0.0).astype(int)

        agg = defaultdict(lambda: {"preds": [], "tgt": []})
        rows = []
        for i, uid in enumerate(ids):
            agg[uid]["preds"].append(int(row_pred[i]))
            agg[uid]["tgt"].append(int(tgt[i]))
            rows.append((uid, row_pred[i], tgt[i]))

        preds = [majority_vote(d["preds"]) for d in agg.values()]
        tgts  = [majority_vote(d["tgt"])  for d in agg.values()]

        mcc = acc = float('nan')
        if len(set(tgts)) > 1:
            mcc = matthews_corrcoef(tgts, preds)
            acc = accuracy_score(tgts, preds)
        return (np.nan, np.nan, np.nan, mcc, acc, rows)

    # ----------------------------------------------------------------- #
    # REGRESSION
    # ----------------------------------------------------------------- #
    preds_np = out.cpu().numpy().flatten()
    tgt_np   = y.cpu().numpy().flatten()

    agg = defaultdict(lambda: {"preds": [], "tgt": []})
    rows = []
    for i, uid in enumerate(ids):
        agg[uid]["preds"].append(preds_np[i])
        agg[uid]["tgt"].append(tgt_np[i])
        rows.append((uid, preds_np[i], tgt_np[i]))

    preds = [np.mean(d["preds"]) for d in agg.values()]
    tgts  = [np.mean(d["tgt"])  for d in agg.values()]

    mse  = float(np.mean((np.array(preds) - np.array(tgts))2))
    r2   = pear = float('nan')
    if len(preds) > 1:
        r2, _ = r2_score(tgts, preds), None
        pear, _ = pearsonr(tgts, preds)

    thr = (args.regression_threshold_log
           if args.data_scale == "log"
           else args.regression_threshold_nonlog)
    pred_cls = (np.array(preds) > thr).astype(int)
    tgt_cls  = (np.array(tgts)  > thr).astype(int)

    mcc = acc = float('nan')
    if len(set(tgt_cls)) > 1:
        mcc = matthews_corrcoef(tgt_cls, pred_cls)
        acc = accuracy_score(tgt_cls, pred_cls)

    return (mse, r2, pear, mcc, acc, rows)

###############################################################################
# CSV output
###############################################################################
def save_predictions(predictions, filename):
    """Write (Label, Predicted, True) rows to *filename*."""
    with open(filename, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Label", "Predicted", "True"])
        wr.writerows(predictions)

###############################################################################
# TRAINING MODE: repeated K-fold
###############################################################################
def run_training_mode(args):
    """
    Core training driver: loops over scramble fractions → repeats → folds,
    trains a model, evaluates on validation split, saves metrics & predictions.
    """
    os.makedirs(args.model_dir, exist_ok=True)

    for frac in args.scramble_fractions:
        frac_str = f"{frac:.2f}".replace(".", "p")
        logging.info(f"=== Training (scr{frac_str}) ===")

        all_metrics, all_rows = [], []

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

                X_trn, y_trn, _   = load_csv_data(trn_csv, args)
                X_val, y_val, ids = load_csv_data(val_csv, args)

                net = Net(X_trn.shape[1], args)
                train_model(net, X_trn, y_trn, X_val, y_val, args)

                # -------- save checkpoint --------------------------------- #
                ckpt = os.path.join(
                    args.model_dir,
                    f"nn_fold_{rep}_{fold}_{args.model_type}_scr{frac_str}.pth"
                )
                T.save(net.state_dict(), ckpt)

                # -------- validation metrics ----------------------------- #
                mse,r2,pear,mcc,acc,rows = evaluate_model(
                    net, X_val, y_val, ids, args
                )
                all_metrics.append(
                    {"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc}
                )
                all_rows.extend(rows)

                # row-level CSV for this fold
                save_predictions(
                    rows,
                    f"{args.output_file}_{args.model_type}_scr{frac_str}"
                    f"_rep{rep}_fold{fold}.csv"
                )

                logging.info(f"[scr{frac_str} rep{rep} fold{fold}] "
                             f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, "
                             f"Pear={fmt_float(pear)}, MCC={fmt_float(mcc)}, "
                             f"Acc={fmt_float(acc)}")

        # ------------- aggregate metrics over all splits ------------------ #
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

        # ------------- aggregate predictions by ID ----------------------- #
        if all_rows:
            agg = defaultdict(lambda: {"preds": [], "tgt": []})
            for uid, pred, tgt in all_rows:
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
    Load saved checkpoints for each repeat / fold and evaluate on the
    held-out test set, then aggregate metrics / predictions.
    """
    os.makedirs(args.model_dir, exist_ok=True)

    for frac in args.scramble_fractions:
        frac_str = f"{frac:.2f}".replace(".", "p")
        tst_csv = os.path.join(
            args.data_dir,
            f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_final.csv"
        )
        if not os.path.isfile(tst_csv):
            logging.warning(f"Test CSV missing: {tst_csv}")
            continue

        logging.info(f"=== Evaluation (scr{frac_str}) ===")
        X_tst, y_tst, ids_tst = load_csv_data(tst_csv, args)

        all_rows, all_metrics = [], []

        for rep in range(args.num_repeats):
            for fold in range(args.kfold):
                ckpt = os.path.join(
                    args.model_dir,
                    f"nn_fold_{rep}_{fold}_{args.model_type}_scr{frac_str}.pth"
                )
                if not os.path.isfile(ckpt):
                    logging.warning(f"Missing ckpt {ckpt}")
                    continue

                net = Net(X_tst.shape[1], args)
                net.load_state_dict(T.load(ckpt, map_location="cpu"))

                mse,r2,pear,mcc,acc,rows = evaluate_model(
                    net, X_tst, y_tst, ids_tst, args
                )
                all_rows.extend(rows)
                all_metrics.append(
                    {"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc}
                )

                logging.info(f"[scr{frac_str} rep{rep} fold{fold}] "
                             f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, "
                             f"Pear={fmt_float(pear)}, MCC={fmt_float(mcc)}, "
                             f"Acc={fmt_float(acc)}")

        # per-row predictions CSV
        save_predictions(
            all_rows,
            f"{args.output_file}_test_{args.model_type}_scr{frac_str}.csv"
        )

        # final averaged metrics CSV
        if all_metrics:
            avg = {}
            for key in all_metrics[0]:
                vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
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

    if   args.mode == 0:
        run_training_mode(args)
    elif args.mode == 1:
        run_evaluation_mode(args)
    else:
        logging.error("--mode must be 0 (train) or 1 (eval).")
        sys.exit(1)


if __name__ == "__main__":
    main()
