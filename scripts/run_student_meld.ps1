# Runs train_student_meld.py -- MELD has a genuine train(1038)/valid(114)/
# test(280) split, so early stopping tracks the REAL valid split every
# epoch (see train_student_meld.py's module docstring). Requires a teacher
# checkpoint for MELD to already exist under output/model/teacher/ -- run
# run_teacher_meld.ps1 first if it doesn't.
#
# Usage (PowerShell): .\run_student_meld.ps1 [-MaskRate <rate>] [-PromptType <type>]
#                      [-LambdaProto <w>] [-LambdaRel <w>] [-Lr <lr>] [-L2 <l2>]
#                      [-Seed <n>] [-RobertaIdx <1-4>]
# MaskRate defaults to 0.1, PromptType defaults to "none". Distillation loss
# is 4 clusters (CE, prob, proto, rel) -- see train_student.py's module
# docstring. CE and prob are fixed at weight 1; only LambdaProto/LambdaRel
# are tunable here. Lr/L2 default to 0.00005/0.0001 (tuned via
# tune_lr_l2_meld.py -- see output/tune/*_lr_l2_sweep_meld.txt). Seed
# defaults to 50. RobertaIdx MUST match whatever run_teacher_meld.ps1 used
# for the checkpoint being loaded (defaults to 1).

param(
    [double]$MaskRate = 0.1,
    [string]$PromptType = "none",
    [double]$LambdaProto = 1.0,
    [double]$LambdaRel = 1.0,
    [double]$Lr = 0.00005,
    [double]$L2 = 0.0001,
    [int]$Seed = 50,
    [int]$RobertaIdx = 1
)

Set-Location -Path (Join-Path $PSScriptRoot '..')

& ".\fsce_net\Scripts\python.exe" train/train_student_meld.py `
    --lstm_hidden_size 128 `
    --graph-layers 1 `
    --hg-layers 1 `
    --lr $Lr `
    --l2 $L2 `
    --dropout 0.1 `
    --prompt-type $PromptType `
    --mask-type "constant-$MaskRate" `
    --lambda-proto $LambdaProto `
    --lambda-rel $LambdaRel `
    --roberta-idx $RobertaIdx `
    --seed $Seed
