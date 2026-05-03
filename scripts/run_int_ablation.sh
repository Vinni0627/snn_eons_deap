#!/bin/bash -x


if [ $# -eq 0 ]; then
    echo "Usage: $0 <dynamic interval>"
    exit 1
fi

INTERVAL=$1
OUT_PATH="results_ablation/obj_${INTERVAL}"

echo "Starting experiment with ${INTERVAL} interval..."
echo "Saving to: ${OUT_PATH}"

# Run from base directory (snn_eons_deap)
python hybrid_experiment.py --int "$INTERVAL" --out_dir "$OUT_PATH" "$@"