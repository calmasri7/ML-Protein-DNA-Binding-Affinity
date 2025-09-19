#!/bin/bash
#
#SBATCH --job-name=rf_final
##SBATCH --account=some_account
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=4

# Activate your environment
source /opt/apps/anaconda/2020.07/bin/activate /data/homezvol1/calmasri/.conda/envs/run_ML

# Abort if any command fails
set -e

MODEL_TYPE=${1:-reg}
SCRAMBLE_FRACTIONS="${2:-0.0}"                # e.g. "0.0 0.25"
SUBSAMPLE_TAGS="${3:-sub100}"                 # e.g. "sub100 sub75 sub50"

read -ra FRAC_ARRAY   <<< "$SCRAMBLE_FRACTIONS"
read -ra SUBTAG_ARRAY <<< "$SUBSAMPLE_TAGS"

# ─── choose the right label column ───────────────────────────────────────────
case "$MODEL_TYPE" in
  bin)    REF_LABEL_COL="improving" ;;
  mclass) REF_LABEL_COL="binding_type" ;;
  reg)    REF_LABEL_COL="bind_avg" ;;
  *) echo "Unknown model type: $MODEL_TYPE"; exit 1 ;;
esac

echo "Model type          : $MODEL_TYPE"
echo "Reference label col : $REF_LABEL_COL"
echo "Scramble fractions  : ${FRAC_ARRAY[*]}"
echo "Subsample tags      : ${SUBTAG_ARRAY[*]}"

NUM_REPEATS=5       # repeated‐K-fold
KFOLD=5

for FRACTION in "${FRAC_ARRAY[@]}"; do
  FRAC_STR=$(echo "$FRACTION" | sed 's/\./p/')

  for SUBTAG in "${SUBTAG_ARRAY[@]}"; do
    echo "============================================================="
    echo "=== scr=${FRACTION}   tag=${SUBTAG} ========================="
    echo "============================================================="

    ############################ 2A) 85/15 split ##############################
    cd Data
	FRAC_NUM=$(echo "$SUBTAG" | sed 's/sub//' | awk '{printf "%.2f", $0/100}')
    python torch_prep_kfold.py \
      --reference_file  ../Inputs/exp_data_all.csv \
      --ref_id_col      sequence \
      --ref_label_col   "$REF_LABEL_COL" \
      --filenames       ../Inputs/rawdat.csv \
      --feature_id_col  sequence \
      --model_type      "$MODEL_TYPE" \
      --random_state    42 \
      --test_percentage 0.2 \
      --prefix          "gbsa" \
      --scramble_fractions "$FRACTION" \
      --subsample_fracs   "$FRAC_NUM" \
      --initial_split
    cd ..

    ############################ 2B) hyper-params #############################
    BEST_JSON="best_hyperparams_${MODEL_TYPE}.json"
    if [ ! -f "$BEST_JSON" ]; then
      echo "Error: $BEST_JSON not found. Did you run hyperopt?"
      exit 1
    fi
    KEEP_LAST_PERCENT=$(jq .keep_last_percent "$BEST_JSON")
    NAVG=$(jq .navg "$BEST_JSON")
    N_EST=$(jq .n_estimators "$BEST_JSON")
    MAX_DEPTH=$(jq .max_depth "$BEST_JSON" | sed 's/"//g')
    MAX_FEATURES=$(jq .max_features "$BEST_JSON" | sed 's/"//g')
    MIN_SPLIT=$(jq .min_samples_split "$BEST_JSON")
    MIN_LEAF=$(jq .min_samples_leaf "$BEST_JSON")

    echo "[Fraction=${FRACTION}] Loaded best hyperparameters from $BEST_JSON:"
    echo "  keep_last_percent = $KEEP_LAST_PERCENT"
    echo "  navg              = $NAVG"
    echo "  n_estimators      = $N_EST"
    echo "  max_depth         = $MAX_DEPTH"
    echo "  max_features      = $MAX_FEATURES"
    echo "  min_samples_split = $MIN_SPLIT"
    echo "  min_samples_leaf  = $MIN_LEAF"

    ############################ 2C) train prep ################################
    cd Data
    python torch_prep_kfold.py \
      --reference_file  ../Inputs/exp_data_all.csv \
      --ref_id_col      sequence \
      --ref_label_col   "$REF_LABEL_COL" \
      --filenames       ../Inputs/rawdat.csv \
      --feature_id_col  sequence \
      --model_type      "$MODEL_TYPE" \
      --keep_last_percent "$KEEP_LAST_PERCENT" \
      --navg              "$NAVG" \
      --random_state      42 \
      --test_percentage   0.2 \
      --prefix            "gbsa" \
      --scramble_fractions "$FRACTION" \
      --subsample_fracs   "$FRAC_NUM" \
      --kfold  $KFOLD \
      --num_repeats $NUM_REPEATS \
      --process train
    cd ..

    ############################ 2D) test prep #################################
    cd Data
    python torch_prep_kfold.py \
      --reference_file  ../Inputs/exp_data_all.csv \
      --ref_id_col      sequence \
      --ref_label_col   "$REF_LABEL_COL" \
      --filenames       ../Inputs/rawdat.csv \
      --feature_id_col  sequence \
      --model_type      "$MODEL_TYPE" \
      --keep_last_percent "$KEEP_LAST_PERCENT" \
      --navg              "$NAVG" \
      --random_state      42 \
      --test_percentage   0.2 \
      --prefix            "gbsa" \
      --scramble_fractions "$FRACTION" \
      --subsample_fracs   "$FRAC_NUM" \
      --kfold  $KFOLD \
      --num_repeats $NUM_REPEATS \
      --process test
    cd ..

    ############################ 2E) RF training ###############################
    python run_model_rf.py \
      --mode 0 \
      --model_type "$MODEL_TYPE" \
      --data_scale log \
      --kfold $KFOLD --num_repeats $NUM_REPEATS \
      --n_estimators "$N_EST" --max_depth "$MAX_DEPTH" \
      --max_features "$MAX_FEATURES" \
      --min_samples_split "$MIN_SPLIT" \
      --min_samples_leaf  "$MIN_LEAF" \
      --ref_id_col sequence --ref_label_col "$REF_LABEL_COL" \
      --model_dir Model --data_dir Data \
      --output_file predictions \
      --scramble_fractions "$FRACTION" \
      --subsample_tags "$SUBTAG"

    ############################ 2F) RF evaluation #############################
    python run_model_rf.py \
      --mode 1 \
      --model_type "$MODEL_TYPE" \
      --data_scale log \
      --kfold $KFOLD --num_repeats $NUM_REPEATS \
      --n_estimators "$N_EST" --max_depth "$MAX_DEPTH" \
      --max_features "$MAX_FEATURES" \
      --min_samples_split "$MIN_SPLIT" \
      --min_samples_leaf  "$MIN_LEAF" \
      --ref_id_col sequence --ref_label_col "$REF_LABEL_COL" \
      --model_dir Model --data_dir Data \
      --output_file predictions \
      --scramble_fractions "$FRACTION" \
      --subsample_tags "$SUBTAG"

    echo "[scr=${FRACTION} | ${SUBTAG}] finished."
  done
done

echo "Complete for scramble fractions: ${FRAC_ARRAY[*]}"
echo "            and subsample tags  : ${SUBTAG_ARRAY[*]}"
