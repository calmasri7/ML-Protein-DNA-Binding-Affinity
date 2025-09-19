#!/bin/bash
#
#SBATCH --job-name=rf_final_noise
##SBATCH --account=some_account
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=4

# ─────────────────────────  Conda env  ──────────────────────────────────────
source /opt/apps/anaconda/2020.07/bin/activate \
       /data/homezvol1/calmasri/.conda/envs/run_ML
set -e   # abort on any error

# ─────────────────────────  CLI args  ───────────────────────────────────────
MODEL_TYPE=${1:-reg}                 # reg | bin | mclass
SCRAMBLE_FRACTIONS="${2:-0.0}"       # e.g. "0.0 0.25"
NOISE_LEVELS="${3:-0.00}"            # e.g. "0.00 0.05 0.10"

read -ra FRAC_ARRAY  <<< "$SCRAMBLE_FRACTIONS"
read -ra NLVL_ARRAY  <<< "$NOISE_LEVELS"

case "$MODEL_TYPE" in
  bin)    REF_LABEL_COL="improving"    ;;
  mclass) REF_LABEL_COL="binding_type" ;;
  reg)    REF_LABEL_COL="bind_avg"     ;;
  *) echo "Unknown model type: $MODEL_TYPE"; exit 1 ;;
esac

echo "Model type          : $MODEL_TYPE"
echo "Label column        : $REF_LABEL_COL"
echo "Scramble fractions  : ${FRAC_ARRAY[*]}"
echo "Noise levels (σ)    : ${NLVL_ARRAY[*]}"

NUM_REPEATS=5
KFOLD=5

# ─────────────────────────  helper: level→tag  ─────────────────────────────
level_to_tag () {                     # 0.05 → noi05
  printf "noi%02d" "$(awk 'BEGIN{printf int('"$1"'*100)}')"
}

# ─────────────────────────  main loops  ─────────────────────────────────────
for FRACTION in "${FRAC_ARRAY[@]}"; do
  FRAC_STR=$(printf "%.2f" "$FRACTION" | sed 's/\./p/')

  for LEVEL in "${NLVL_ARRAY[@]}"; do
    TAG=$(level_to_tag "$LEVEL")

    echo "============================================================="
    echo "===  scr=${FRACTION}   noise=${LEVEL} (${TAG})  ============="
    echo "============================================================="

    ############################################################################
    # 1) INITIAL 85/15 SPLIT  (adds sequence-scramble, NO noise yet)
    ############################################################################
    cd Data
    python torch_prep_kfold.py \
        --reference_file  ../Inputs/exp_data_all.csv \
        --ref_id_col      sequence \
        --ref_label_col   "$REF_LABEL_COL" \
        --filenames       ../Inputs/rawdat.csv \
        --feature_id_col  sequence \
        --model_type      "$MODEL_TYPE" \
        --random_state    42 \
        --test_percentage 0.20 \
        --prefix          gbsa \
        --scramble_fractions "$FRACTION" \
        --noise_levels       "$LEVEL" \
        --initial_split
    cd ..

    ############################################################################
    # 2) BEST HYPER-PARAMS  (from your JSON)
    ############################################################################
    BEST_JSON="best_hyperparams_${MODEL_TYPE}.json"
    [[ -f $BEST_JSON ]] || { echo "Missing $BEST_JSON"; exit 1; }

    KEEP_LAST_PERCENT=$(jq .keep_last_percent  "$BEST_JSON")
    NAVG=$(jq .navg                          "$BEST_JSON")
    N_EST=$(jq .n_estimators                 "$BEST_JSON")
    MAX_DEPTH=$(jq -r .max_depth             "$BEST_JSON")
    MAX_FEATURES=$(jq -r .max_features       "$BEST_JSON")
    MIN_SPLIT=$(jq .min_samples_split        "$BEST_JSON")
    MIN_LEAF=$(jq .min_samples_leaf          "$BEST_JSON")

    ############################################################################
    # 3) TRAIN-SET PREP  (adds label-noise + CV CSVs)
    ############################################################################
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
        --test_percentage   0.20 \
        --prefix            gbsa \
        --scramble_fractions "$FRACTION" \
        --noise_levels       "$LEVEL" \
        --kfold  $KFOLD  --num_repeats $NUM_REPEATS \
        --process train
    cd ..

    ############################################################################
    # 4) TEST-SET PREP  (no noise; uses stats from clean data)
    ############################################################################
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
        --test_percentage   0.20 \
        --prefix            gbsa \
        --scramble_fractions "$FRACTION" \
        --noise_levels       "$LEVEL" \
        --process test
    cd ..

    ############################################################################
    # 5) RANDOM-FOREST TRAIN
    ############################################################################
    python run_model_rf.py \
        --mode 0 \
        --model_type "$MODEL_TYPE" \
        --data_scale log \
        --kfold $KFOLD --num_repeats $NUM_REPEATS \
        --n_estimators "$N_EST"  --max_depth "$MAX_DEPTH" \
        --max_features "$MAX_FEATURES" \
        --min_samples_split "$MIN_SPLIT" \
        --min_samples_leaf  "$MIN_LEAF" \
        --ref_id_col sequence --ref_label_col "$REF_LABEL_COL" \
        --model_dir Model --data_dir Data \
        --output_file predictions \
        --scramble_fractions "$FRACTION" \
        --noise_tags "$TAG"

    ############################################################################
    # 6) RANDOM-FOREST EVAL
    ############################################################################
    python run_model_rf.py \
        --mode 1 \
        --model_type "$MODEL_TYPE" \
        --data_scale log \
        --kfold $KFOLD --num_repeats $NUM_REPEATS \
        --n_estimators "$N_EST"  --max_depth "$MAX_DEPTH" \
        --max_features "$MAX_FEATURES" \
        --min_samples_split "$MIN_SPLIT" \
        --min_samples_leaf  "$MIN_LEAF" \
        --ref_id_col sequence --ref_label_col "$REF_LABEL_COL" \
        --model_dir Model --data_dir Data \
        --output_file predictions \
        --scramble_fractions "$FRACTION" \
        --noise_tags "$TAG"

    echo "[scr=${FRACTION} | ${TAG}]  complete."
  done
done

echo "Finished: scramble fractions  = ${FRAC_ARRAY[*]}"
echo "          noise levels (σ)     = ${NLVL_ARRAY[*]}"
