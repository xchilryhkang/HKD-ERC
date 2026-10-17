#!/bin/bash
# Runs train_teacher.py with the hyperparameters found by the tune_*.py
# sweeps (see output/tune/*.txt for the full sweep logs this came from):
#
#   lstm_hidden_size=128 (-> hidden_dim=256)   tune_hidden_dim.py
#   lr=0.0001                                   tune_lr_l2.py
#   l2=0.00001 (default, unchanged)             tune_lr_l2.py
#   graph-layers=1                              tune_layers.py
#   hg-layers=1                                 tune_layers.py
#   n-graph-heads=4 (default, confirmed best)   tune_heads.py
#   windowp=windowf=3 (default, confirmed best) tune_window.py
#   dropout=0.1 (best so far; 0.0/0.05 not yet  tune_dropout.py
#                tested — the trend hadn't
#                plateaued, see that sweep's summary)
#
# Run from the code/ directory: ./run_teacher.sh [dataset]
# dataset defaults to IEMOCAPSix if not given.

set -e

cd "$(dirname "$0")"

DATASET="${1:-IEMOCAPSix}"

./fsce_net/Scripts/python.exe -u train_teacher.py \
    --dataset "$DATASET" \
    --lstm_hidden_size 128 \
    --lr 0.0001 \
    --l2 0.00001 \
    --graph-layers 1 \
    --hg-layers 1 \
    --dropout 0.1
