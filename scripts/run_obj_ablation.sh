#!/bin/bash -x


if [ $# -eq 0 ]; then
    echo "Usage: $0 <n_objects>"
    exit 1
fi

OBJ_COUNT=$1
OUT_PATH="results_ablation/obj_${OBJ_COUNT}"

echo "Starting experiment with ${OBJ_COUNT} objects..."
echo "Saving to: ${OUT_PATH}"

# Run from base directory (snn_eons_deap)
python3 hybrid_experiment.py --obj "$OBJ_COUNT" --out_dir "$OUT_PATH" 