#!/bin/bash
# Runs train_student_iemocap.py for IEMOCAPSix -- 5-fold leave-one-session-
# out CV, best checkpoint = best session-out F1 (early stopping tracks the
# held-out session directly every epoch; see train_student_iemocap.py's
# module docstring for why there's no separate validation session under
# this protocol). Requires a teacher checkpoint for IEMOCAPSix to already
# exist under output/model/teacher/ -- run run_teacher_iemocap6.sh first
# if it doesn't.
#
# Run from the code/ directory, any flag optional -- omitted ones fall back
# to their default, passed ones override it:
#   ./run_student_iemocap6.sh [--mask-rate 0.1] [--prompt-type none]
#       [--lr 0.0002] [--l2 0.00001] [--optimizer adamw] [--dropout 0.1] [--seed 50]
#       [--lambda-proto 1.0] [--lambda-rel 1.0]
# Distillation loss is 4 clusters (CE, prob, proto, rel) -- see
# train_student.py's module docstring. CE and prob are fixed at weight 1;
# only --lambda-proto/--lambda-rel are tunable here. Lr/L2/optimizer default
# to the teacher-tuned winner from tune_lr_l2_iemocap.py (no separate
# student sweep has been run).

set -e

cd "$(dirname "$0")/.."

MASK_RATE=0.1
PROMPT_TYPE=none
LR=0.0002
L2=0.00001
OPTIMIZER=adamw
DROPOUT=0.1
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
    --lambda-proto "$LAMBDA_PROTO" \
    --lambda-rel "$LAMBDA_REL" \
    --seed "$SEED"
