#!/usr/bin/env python


"""
run_model.py

This script trains or evaluates a PyTorch neural network (MLP) using repeated K-fold splits
and optionally multiple scramble fractions.

Modes:
------
1) mode=0 (training):
   - For each fraction in --scramble_fractions,
     - For each repeat_idx in [0..num_repeats-1] and fold_idx in [0..kfold-1]:
         * Load the CSV files:
             {prefix}_{model_type}_scrFRAC_trn_{repeat_idx}_{fold_idx}.csv
             {prefix}_{model_type}_scrFRAC_val_{repeat_idx}_{fold_idx}.csv
         * Instantiate the MLP with arguments:
             --hidden_size, --hidden_layers, --dropout_input_output, --dropout_hidden
         * Train the network (with optional early stopping);
           save best model checkpoint to:
             nn_fold_{repeat_idx}_{fold_idx}_{model_type}_scrFRAC.pth
         * Evaluate on the validation split → collect metrics & row-level predictions.

2) mode=1 (evaluation):
   - For each fraction in --scramble_fractions:
       * Load the test CSV:
           {prefix}_{model_type}_scrFRAC_tst_final.csv
       * For each repeat_idx and fold_idx:
           - Load the saved model:
               nn_fold_{repeat_idx}_{fold_idx}_{model_type}_scrFRAC.pth
           - Evaluate on the test set → gather metrics & row-level predictions
       * Write out per-row predictions and aggregated metrics.

Usage Examples:
---------------
# Training with two scramble fractions, 5-fold CV repeated 3 times, early stopping enabled
python run_model_nn.py \
  --mode 0 \
  --model_type reg \
  --scramble_fractions 0.0 0.25 \
  --prefix gbsa \
  --kfold 5 \
  --num_repeats 3 \
  --hidden_size 44 \
  --hidden_layers 3 \
  --dropout_input_output 0.1 \
  --dropout_hidden 0.1 \
  --lrn_rate 1e-4 \
  --wt_decay 1e-4 \
  --use_early_stopping \
  --patience 50

# Evaluation on those same fractions
python run_model_nn.py \
  --mode 1 \
  --model_type reg \
  --scramble_fractions 0.0 0.25 \
  --prefix gbsa \
  --kfold 5 \
  --num_repeats 3 \
  --hidden_size 44 \
  --hidden_layers 3 \
  --dropout_input_output 0.1 \
  --dropout_hidden 0.1 \
  --lrn_rate 1e-4 \
  --wt_decay 1e-4
"""

###############################################################################
# Imports
###############################################################################
import argparse            
import sys, os, csv         
import logging               
from typing import Tuple, List, Any
from collections import defaultdict

# Numerical / ML stack
import numpy as np
import pandas as pd
import torch as T           
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,          
    r2_score                
)
from scipy.stats import pearsonr  

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
    """
    Format a float with 4 digits after the decimal.
    If `x` is NaN or not a float → return the string 'NaN'.
    """
    return f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "NaN"


def majority_vote(values: List[int]) -> int:
    """
    Return the most common integer in *values*.
    Ties are broken by `pandas.Series.mode()`, which returns the
    first encountered mode.
    """
    return int(pd.Series(values).mode()[0])

###############################################################################
# Argument Parsing
###############################################################################
def parse_args():
    """
    Build and parse the command-line interface.
    Everything the user can tweak lives here – no hard-coded paths.
    """
    parser = argparse.ArgumentParser(
        description="Neural-network training / evaluation script"
    )

    # ------------------------ core flags ---------------------------------- #
    parser.add_argument("--mode", type=int, default=0,
                        help="0 → training mode, 1 → evaluation mode.")
    parser.add_argument("--model_type", type=str, default="reg",
                        choices=["reg", "bin", "mclass"],
                        help="'reg' (regression), 'bin' (binary cls), 'mclass'.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes for multi-class tasks.")
    parser.add_argument("--data_scale", type=str, default="log",
                        choices=["log", "nonlog"],
                        help="Determines threshold for ΔΔG classification.")

    # ------------------------ cross-validation ---------------------------- #
    parser.add_argument("--kfold", type=int, default=5,
                        help="#folds for K-fold CV.")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Repeat the K-fold split `n` times.")
    parser.add_argument("--scramble_fractions", type=float, nargs="+",
                        default=[0.0],
                        help="Label-scrambling fractions for leakage tests.")

    # ------------------------ hyper-parameters ---------------------------- #
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--lrn_rate", type=float, default=1e-4)
    parser.add_argument("--wt_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_input_output", type=float, default=0.1)
    parser.add_argument("--dropout_hidden", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=44)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=50)

    # ------------- thresholds used to binarize regression outputs --------- #
    parser.add_argument("--regression_threshold_log", type=float, default=0.0)
    parser.add_argument("--regression_threshold_nonlog", type=float, default=1.0)

    # ------------------------ IO paths ------------------------------------ #
    parser.add_argument("--model_dir", type=str, default="Model/")
    parser.add_argument("--data_dir", type=str, default="Data/")
    parser.add_argument("--output_file", type=str, default="predictions")
    parser.add_argument("--prefix", type=str, default="gbsa")

    # --------------- column names inside the data CSV --------------------- #
    parser.add_argument("--ref_id_col", type=str, default="sequence")
    parser.add_argument("--ref_label_col", type=str, default="label")

    return parser.parse_args()

###############################################################################
# Data Loading
###############################################################################
def load_csv_data(csv_file: str, args) -> Tuple[T.Tensor, T.Tensor, List[Any]]:
    """
    Read a CSV file and convert to PyTorch tensors.
    *   ID   column → kept as Python list for aggregation.
    * target column → 1-D float32 tensor (with extra dim so shape = [N,1]).
    * feature cols → float32 tensor  (shape = [N, #features])
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Data file not found: {csv_file}")

    df = pd.read_csv(csv_file, header=0)
    id_col, label_col = args.ref_id_col, args.ref_label_col

    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV {csv_file} must contain '{id_col}' and '{label_col}' columns."
        )

    # treat every other column as numeric feature
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]

    ids      = df[id_col].tolist()
    features = T.tensor(df[feature_cols].values.astype(np.float32))
    targets  = T.tensor(df[label_col].values.astype(np.float32)).unsqueeze(1)
    return features, targets, ids

###############################################################################
# Neural Network Definition
###############################################################################
class Net(T.nn.Module):
    """
    Simple fully-connected network:
    [input] → ReLU → (dropout) → hidden × (L-1) → ReLU → (dropout) → out
    Output layer:
       * regression / binary cls → 1 neuron
       * multi-class             → `num_classes` neurons (logits)
    """
    def __init__(self, input_dim: int, args):
        super().__init__()

        # ---------------- network architecture --------------------------- #
        out_dim = args.num_classes if args.model_type == "mclass" else 1

        layers = []
        # first layer
        layers.append(T.nn.Linear(input_dim, args.hidden_size))
        # additional hidden layers
        for _ in range(args.hidden_layers - 1):
            layers.append(T.nn.Linear(args.hidden_size, args.hidden_size))
        # final output
        layers.append(T.nn.Linear(args.hidden_size, out_dim))
        self.layers = T.nn.ModuleList(layers)

        # activation / dropout
        self.act          = T.nn.ReLU()
        self.dropout_io   = T.nn.Dropout(args.dropout_input_output)
        self.dropout_hid  = T.nn.Dropout(args.dropout_hidden)

        self._init_weights()   # Xavier init
        self.args = args

    def _init_weights(self):
        """Xavier-uniform for weights, zeros for biases."""
        for layer in self.layers:
            T.nn.init.xavier_uniform_(layer.weight)
            T.nn.init.zeros_(layer.bias)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Forward pass with dropout after first & hidden layers.
        """
        x = self.dropout_io(self.act(self.layers[0](x)))
        for layer in self.layers[1:-1]:
            x = self.dropout_hid(self.act(layer(x)))
        return self.layers[-1](x)   # logits / regression value

###############################################################################
# Loss Function 
###############################################################################
def get_loss_function(args):
    """Return the appropriate criterion given task type."""
    if args.model_type == "reg":
        return T.nn.MSELoss()
    elif args.model_type == "bin":
        return T.nn.BCEWithLogitsLoss()
    elif args.model_type == "mclass":
        return T.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown model_type '{args.model_type}'")

###############################################################################
# Early-Stopping Helper
###############################################################################
class EarlyStopper:
    """
    Monitor a validation metric (`min` or `max`) and stop training if it
    hasn't improved for `patience` epochs. Stores the best model weights.
    """
    def __init__(self, patience=20, mode="min"):
        self.patience   = patience
        self.mode       = mode
        self.best_value = None
        self.counter    = 0
        self.should_stop = False
        self.best_model_state = None

    def check(self, current_value, model_state_dict):
        """
        Update internal state; set `should_stop` if patience exceeded.
        """
        if self.best_value is None:
            # first observation
            self.best_value = current_value
            self.best_model_state = model_state_dict
            return

        improved = (
            current_value < self.best_value if self.mode == "min"
            else current_value > self.best_value
        )

        if improved:
            self.best_value = current_value
            self.best_model_state = model_state_dict
            self.counter = 0           # reset patience
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

###############################################################################
# Training Loop
###############################################################################
def train_model(net: Net,
                trn_feat: T.Tensor, trn_tgt: T.Tensor,
                val_feat: T.Tensor, val_tgt: T.Tensor,
                args):
    """
    minibatch-free training loop (full-batch gradient descent).
    Early stopping optional. Metrics printed every 10 % of epochs.
    """
    criterion = get_loss_function(args)
    optimizer = T.optim.Adam(
        net.parameters(),
        lr=args.lrn_rate,
        weight_decay=args.wt_decay
    )

    # reshape targets for classification tasks
    if args.model_type == "mclass":
        trn_tgt = trn_tgt.view(-1).long()
        val_tgt = val_tgt.view(-1).long()

    early_stopper = (
        EarlyStopper(patience=args.patience, mode="min")
        if args.use_early_stopping else None
    )
    log_every = max(1, args.max_epochs // 10)

    for ep in range(args.max_epochs):
        optimizer.zero_grad()

        preds = net(trn_feat)
        loss  = criterion(preds, trn_tgt)
        loss.backward()
        optimizer.step()

        # periodic log
        if (ep + 1) % log_every == 0:
            logging.info(f"[Epoch {ep+1}/{args.max_epochs}] "
                         f"train-loss={loss.item():.4f}")

        # ----------------- early stopping ------------------------------ #
        if early_stopper:
            net.eval()
            with T.no_grad():
                val_preds = net(val_feat)
                val_loss  = criterion(val_preds, val_tgt).item()
            net.train()

            early_stopper.check(val_loss, net.state_dict())
            if early_stopper.should_stop:
                logging.info(f"Early stopping at epoch {ep+1} "
                             f"(best val-loss={early_stopper.best_value:.4f})")
                net.load_state_dict(early_stopper.best_model_state)
                break

###############################################################################
# Evaluation / Prediction
###############################################################################
def evaluate_model(net: Net,
                   features: T.Tensor,
                   targets: T.Tensor,
                   ids: List[Any],
                   args):
    """
    Forward-prop through the network and compute metrics.  
    Multiple rows may share the same ID (e.g., snapshots of same complex).
    Rows → aggregated to per-ID predictions:

    * Regression  → average of predictions.
    * Classification → majority vote (or summed logits for mclass).
    """
    net.eval()
    with T.no_grad():
        out = net(features)

    # ------------------------------------------------------------------- #
    # MULTI-CLASS CLASSIFICATION
    # ------------------------------------------------------------------- #
    if args.model_type == "mclass":
        logits = out.detach().cpu().numpy()            # shape [N, C]
        tgt_np = targets.view(-1).cpu().numpy()        # [N]
        row_preds = np.argmax(logits, axis=1)          # predicted class per row

        # aggregate predictions / targets by sequence ID
        aggregator = defaultdict(lambda: {"logits": [], "tgt": []})
        row_level_data = []                            # keep per-row CSV

        for i, uid in enumerate(ids):
            aggregator[uid]["logits"].append(logits[i])
            aggregator[uid]["tgt"].append(int(tgt_np[i]))
            row_level_data.append((uid, row_preds[i], tgt_np[i]))

        agg_preds, agg_tgts = [], []
        for uid, d in aggregator.items():
            summed_logits = np.sum(d["logits"], axis=0)   # vote by summing
            agg_preds.append(int(np.argmax(summed_logits)))
            agg_tgts.append(majority_vote(d["tgt"]))

        # compute MCC / accuracy only if >1 class present in ground truth
        mcc, acc = float('nan'), float('nan')
        if len(set(agg_tgts)) > 1:
            mcc = matthews_corrcoef(agg_tgts, agg_preds)
            acc = accuracy_score(agg_tgts, agg_preds)

        # Placeholder NaNs for regression metrics
        return (np.nan, np.nan, np.nan, mcc, acc, row_level_data)

    # ------------------------------------------------------------------- #
    # BINARY CLASSIFICATION
    # ------------------------------------------------------------------- #
    elif args.model_type == "bin":
        logits = out.detach().cpu().numpy().flatten()  # raw scores
        tgt_np = targets.detach().cpu().numpy().flatten()
        row_preds = (logits > 0.0).astype(int)

        aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
        row_level_data = []

        for i, uid in enumerate(ids):
            aggregator[uid]["preds"].append(int(row_preds[i]))
            aggregator[uid]["tgt"].append(int(tgt_np[i]))
            row_level_data.append((uid, int(row_preds[i]), int(tgt_np[i])))

        agg_preds = [majority_vote(d["preds"]) for d in aggregator.values()]
        agg_tgts  = [majority_vote(d["tgt"])  for d in aggregator.values()]

        mcc, acc = float('nan'), float('nan')
        if len(set(agg_tgts)) > 1:
            mcc = matthews_corrcoef(agg_tgts, agg_preds)
            acc = accuracy_score(agg_tgts,  agg_preds)

        return (np.nan, np.nan, np.nan, mcc, acc, row_level_data)

    # ------------------------------------------------------------------- #
    # REGRESSION
    # ------------------------------------------------------------------- #
    else:
        preds_np = out.detach().cpu().numpy().flatten()
        tgt_np   = targets.detach().cpu().numpy().flatten()

        aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
        row_level_data = []

        for i, uid in enumerate(ids):
            aggregator[uid]["preds"].append(preds_np[i])
            aggregator[uid]["tgt"].append(tgt_np[i])
            row_level_data.append((uid, preds_np[i], tgt_np[i]))

        # mean over rows for each ID
        agg_preds = [np.mean(d["preds"]) for d in aggregator.values()]
        agg_tgts  = [np.mean(d["tgt"])   for d in aggregator.values()]

        # classic regression metrics
        mse  = float(np.mean((np.array(agg_preds) - np.array(agg_tgts))**2))
        r2   = float('nan')
        pear = float('nan')
        if len(agg_preds) > 1:
            r2   = float(r2_score(agg_tgts, agg_preds))
            pear,_ = pearsonr(agg_tgts, agg_preds)

        # Additionally compute MCC / accuracy by binarizing ΔΔG
        thr = (args.regression_threshold_log
               if args.data_scale == "log"
               else args.regression_threshold_nonlog)

        pred_cls = (np.array(agg_preds) > thr).astype(int)
        tgt_cls  = (np.array(agg_tgts) > thr).astype(int)

        mcc, acc = float('nan'), float('nan')
        if len(set(tgt_cls)) > 1:
            mcc = matthews_corrcoef(tgt_cls, pred_cls)
            acc = accuracy_score(tgt_cls,  pred_cls)

        return (mse, r2, pear, mcc, acc, row_level_data)

###############################################################################
# CSV Helper
###############################################################################
def save_predictions(predictions, filename: str):
    """
    Write a CSV with three columns: *Label*, *Predicted*, *True*.
    `predictions` is an iterable of tuples (label, pred, tgt).
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
    Training workflow:
      for each scramble-fraction
        for each repeat
          for each fold
             * load train / val CSV
             * fit network, save .pth
             * evaluate on val → store metrics / row-predictions
    After loops:
      * write overall metrics CSV
      * write aggregated per-ID prediction CSV
    """
    os.makedirs(args.model_dir, exist_ok=True)

    for fraction in args.scramble_fractions:
        frac_str = f"{fraction:.2f}".replace(".", "p")
        logging.info(f"=== Training (scr{frac_str}) ===")

        all_metrics, all_predictions = [], []

        # ------------------- iterate repeats / folds -------------------- #
        for repeat_idx in range(args.num_repeats):
            for fold_idx in range(args.kfold):
                # -------- locate CSVs produced by external data-prep ------ #
                trn_file = os.path.join(
                    args.data_dir,
                    f"{args.prefix}_{args.model_type}_scr{frac_str}_trn_{repeat_idx}_{fold_idx}.csv"
                )
                val_file = os.path.join(
                    args.data_dir,
                    f"{args.prefix}_{args.model_type}_scr{frac_str}_val_{repeat_idx}_{fold_idx}.csv"
                )

                if not (os.path.isfile(trn_file) and os.path.isfile(val_file)):
                    logging.warning(f"[scr{frac_str}] Missing CSV "
                                    f"(rep={repeat_idx}, fold={fold_idx})")
                    continue

                # ------------------- load data --------------------------- #
                X_trn, y_trn, ids_trn = load_csv_data(trn_file, args)
                X_val, y_val, ids_val = load_csv_data(val_file, args)

                # ------------------- build & train ----------------------- #
                net = Net(X_trn.shape[1], args)
                logging.info(f"[scr{frac_str}] Training rep={repeat_idx}, fold={fold_idx}")
                train_model(net, X_trn, y_trn, X_val, y_val, args)

                # save model checkpoint
                model_path = os.path.join(
                    args.model_dir,
                    f"nn_fold_{repeat_idx}_{fold_idx}_{args.model_type}_scr{frac_str}.pth"
                )
                T.save(net.state_dict(), model_path)

                # ------------------- validation metrics ------------------ #
                mse,r2,pear,mcc,acc,fold_preds = evaluate_model(
                    net, X_val, y_val, ids_val, args
                )
                all_metrics.append(
                    {"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc}
                )
                all_predictions.extend(fold_preds)

                # save per-fold prediction CSV
                fold_csv = (f"{args.output_file}_{args.model_type}_scr{frac_str}"
                            f"_rep{repeat_idx}_fold{fold_idx}.csv")
                save_predictions(fold_preds, fold_csv)

                logging.info(f"[scr{frac_str} rep{repeat_idx} fold{fold_idx}] "
                             f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, "
                             f"Pear={fmt_float(pear)}, MCC={fmt_float(mcc)}, "
                             f"Acc={fmt_float(acc)}")

        # ------------------ aggregate metrics across folds -------------- #
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0]:
                vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
                avg_metrics[key] = float(np.mean(vals)) if vals else float('nan')

            logging.info(f"[scr{frac_str}] FINAL average validation metrics:")
            for k,v in avg_metrics.items():
                logging.info(f"  {k} = {fmt_float(v)}")

            # write metrics CSV
            with open(f"final_metrics_{args.model_type}_trn_scr{frac_str}.csv",
                      "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["MSE","R2","Pear","MCC","Accuracy"])
                w.writeheader()
                w.writerow({k: fmt_float(v) for k,v in avg_metrics.items()})

        # ------------------ aggregate predictions by ID ----------------- #
        if all_predictions:
            aggregator = defaultdict(lambda: {"preds": [], "tgt": []})
            for uid, pred, tgt in all_predictions:
                aggregator[uid]["preds"].append(pred)
                aggregator[uid]["tgt"].append(tgt)

            labels, preds, tgts = [], [], []
            for uid, d in aggregator.items():
                labels.append(uid)
                if args.model_type in ["bin", "mclass"]:
                    preds.append(majority_vote(d["preds"]))
                    tgts.append(majority_vote(d["tgt"]))
                else:
                    preds.append(np.mean(d["preds"]))
                    tgts.append(np.mean(d["tgt"]))

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
    Evaluate previously trained models on the held-out test set.
    Aggregates predictions/metrics over repeats & folds.
    """
    os.makedirs(args.model_dir, exist_ok=True)

    for fraction in args.scramble_fractions:
        frac_str = f"{fraction:.2f}".replace(".", "p")
        test_csv = os.path.join(
            args.data_dir,
            f"{args.prefix}_{args.model_type}_scr{frac_str}_tst_final.csv"
        )
        if not os.path.isfile(test_csv):
            logging.warning(f"[scr{frac_str}] Test CSV missing ({test_csv})")
            continue

        logging.info(f"=== Evaluation (scr{frac_str}) ===")
        X_tst, y_tst, ids_tst = load_csv_data(test_csv, args)

        predictions, metrics = [], []

        for repeat_idx in range(args.num_repeats):
            for fold_idx in range(args.kfold):
                model_path = os.path.join(
                    args.model_dir,
                    f"nn_fold_{repeat_idx}_{fold_idx}_{args.model_type}_scr{frac_str}.pth"
                )
                if not os.path.isfile(model_path):
                    logging.warning(f"[scr{frac_str}] Missing model "
                                    f"(rep={repeat_idx}, fold={fold_idx})")
                    continue

                net = Net(X_tst.shape[1], args)
                net.load_state_dict(T.load(model_path, map_location="cpu"))

                mse,r2,pear,mcc,acc,fold_preds = evaluate_model(
                    net, X_tst, y_tst, ids_tst, args
                )
                predictions.extend(fold_preds)
                metrics.append(
                    {"MSE": mse, "R2": r2, "Pear": pear, "MCC": mcc, "Accuracy": acc}
                )

                logging.info(f"[scr{frac_str} rep{repeat_idx} fold{fold_idx}] "
                             f"MSE={fmt_float(mse)}, R2={fmt_float(r2)}, "
                             f"Pear={fmt_float(pear)}, MCC={fmt_float(mcc)}, "
                             f"Acc={fmt_float(acc)}")

        # ----------- save row-level predictions ------------------------- #
        save_predictions(
            predictions,
            f"{args.output_file}_test_{args.model_type}_scr{frac_str}.csv"
        )

        # ----------- aggregate & save test metrics ---------------------- #
        if metrics:
            avg_metrics = {}
            for key in metrics[0]:
                vals = [m[key] for m in metrics if not np.isnan(m[key])]
                avg_metrics[key] = float(np.mean(vals)) if vals else float('nan')

            for k,v in avg_metrics.items():
                logging.info(f"[scr{frac_str}] TEST {k} = {fmt_float(v)}")

            with open(f"final_metrics_{args.model_type}_tst_scr{frac_str}.csv",
                      "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["MSE","R2","Pear","MCC","Accuracy"])
                w.writeheader()
                w.writerow({k: fmt_float(v) for k,v in avg_metrics.items()})

###############################################################################
# Main
###############################################################################
def main():
    args = parse_args()

    # ensure dirs exist (they may already)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir,  exist_ok=True)

    if   args.mode == 0:
        run_training_mode(args)
    elif args.mode == 1:
        run_evaluation_mode(args)
    else:
        logging.error("`--mode` must be 0 (train) or 1 (eval).")
        sys.exit(1)


if __name__ == "__main__":
    main()