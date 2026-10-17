#!/bin/bash
# Runs train_teacher_iemocap.py for IEMOCAPSix -- 5-fold leave-one-session-
# out CV, best checkpoint = best session-out F1 (early stopping tracks the
# held-out session directly every epoch; see train_teacher_iemocap.py's
# module docstring for why there's no separate validation session under
# this protocol).
#
# Run from the code/ directory, any flag optional -- omitted ones fall back
# to their default, passed ones override it:
#   ./run_teacher_iemocap6.sh [--lr 0.0002] [--l2 0.00001] [--optimizer adamw] [--dropout 0.1] [--seed 50]

set -e

cd "$(dirname "$0")"

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

./fsce_net/Scripts/python.exe -u train_teacher_iemocap.py \
    --dataset IEMOCAPSix \
    --lstm_hidden_size 128 \
    --lr "$LR" \
    --l2 "$L2" \
    --optimizer "$OPTIMIZER" \
    --graph-layers 1 \
    --hg-layers 1 \
    --dropout "$DROPOUT" \
    --seed "$SEED"
