#!/bin/bash
# Needs a teacher checkpoint under output/model/teacher/ first.

set -e

cd "$(dirname "$0")/.."

MASK_RATE=0.1
PROMPT_TYPE=none
LR=0.0002
L2=0.00001
OPTIMIZER=adamw
DROPOUT=0.1
LAMBDA_PROB=1.0
LAMBDA_PROTO=1.0
LAMBDA_REL=1.0
SEED=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mask-rate) MASK_RATE="$2"; shift 2 ;;
        --prompt-type) PROMPT_TYPE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --l2) L2="$2"; shift 2 ;;
        --optimizer) OPTIMIZER="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --lambda-prob) LAMBDA_PROB="$2"; shift 2 ;;
        --lambda-proto) LAMBDA_PROTO="$2"; shift 2 ;;
        --lambda-rel) LAMBDA_REL="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

./fsce_net/Scripts/python.exe -u train/train_student_iemocap.py \
    --dataset IEMOCAPSix \
    --lstm_hidden_size 128 \
    --graph-layers 1 \
    --hg-layers 1 \
    --lr "$LR" \
    --l2 "$L2" \
    --optimizer "$OPTIMIZER" \
    --dropout "$DROPOUT" \
    --prompt-type "$PROMPT_TYPE" \
    --mask-type "constant-$MASK_RATE" \
    --lambda-prob "$LAMBDA_PROB" \
    --lambda-proto "$LAMBDA_PROTO" \
    --lambda-rel "$LAMBDA_REL" \
    --seed "$SEED"
