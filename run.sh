#!/usr/bin/env bash
# PACT: the whole sweep, both GPUs, detached.
#
#   ./run.sh              two GPUs (0,1), default config
#   ./run.sh 0,1,2,3      four GPUs
#   ./run.sh 0 configs/server_9b_24gb_qlora.json
#
# Progress: tail -f train.log     Per-job detail: .cache/runs/*/*/seed-*/worker.log
# When it finishes, open results/SUMMARY.md
set -euo pipefail
cd "$(dirname "$0")"

DEVICES="${1:-0,1}"
CONFIG="${2:-configs/server_2x48gb.json}"

export CUDA_VISIBLE_DEVICES="$DEVICES"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

nohup python train_all.py --config "$CONFIG" > train.log 2>&1 &
echo "started pid $! on GPUs $DEVICES with $CONFIG"
echo "follow it with: tail -f $(pwd)/train.log"
echo "when it finishes: $(pwd)/results/SUMMARY.md"
