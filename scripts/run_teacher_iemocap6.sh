#!/bin/bash

set -e

cd "$(dirname "$0")/.."

LR=0.0002
L2=0.00001
OPTIMIZER=adamw
DROPOUT=0.1
SEED=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr) LR="$2"; shift 2 ;;
        --l2) L2="$2"; shift 2 ;;
        --optimizer) OPTIMIZER="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

./fsce_net/Scripts/python.exe -u train/train_teacher_iemocap.py \
    --dataset IEMOCAPSix \
    --lstm_hidden_size 128 \
    --lr "$LR" \
    --l2 "$L2" \
    --optimizer "$OPTIMIZER" \
    --graph-layers 1 \
    --hg-layers 1 \
    --dropout "$DROPOUT" \
    --seed "$SEED"
