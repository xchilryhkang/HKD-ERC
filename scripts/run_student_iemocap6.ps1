# Runs train_student_iemocap.py for IEMOCAPSix -- 5-fold leave-one-session-
# out CV, best checkpoint = best session-out F1 (early stopping tracks the
# held-out session directly every epoch; see train_student_iemocap.py's
# module docstring for why there's no separate validation session under
# this protocol). Requires a teacher checkpoint for IEMOCAPSix to already
# exist under output/model/teacher/ -- run run_teacher_iemocap6.ps1 first
# if it doesn't.
#
# Usage (PowerShell): .\run_student_iemocap6.ps1 [-MaskRate <rate>] [-PromptType <type>]
#                      [-LambdaProto <w>] [-LambdaRel <w>] [-Lr <lr>] [-L2 <l2>] [-Optimizer <adam|adamw>] [-Seed <n>]
# MaskRate defaults to 0.1, PromptType defaults to "none". Distillation loss
# is 4 clusters (CE, prob, proto, rel) -- see train_student.py's module
# docstring. CE and prob are fixed at weight 1; only LambdaProto/LambdaRel
# are tunable here (each defaults to 1.0). Lr/L2/Optimizer default to
# 0.0002/0.00001/adamw (teacher-tuned winner from tune_lr_l2_iemocap.py;
# no separate student sweep has been run). Seed defaults to 50.

param(
    [double]$MaskRate = 0.1,
    [string]$PromptType = "none",
    [double]$LambdaProto = 1.0,
    [double]$LambdaRel = 1.0,
    [double]$Lr = 0.0002,
    [double]$L2 = 0.00001,
    [string]$Optimizer = 'adamw',
    [int]$Seed = 50
)

Set-Location -Path (Join-Path $PSScriptRoot '..')

& ".\fsce_net\Scripts\python.exe" train/train_student_iemocap.py `
    --dataset IEMOCAPSix `
    --lstm_hidden_size 128 `
    --graph-layers 1 `
    --hg-layers 1 `
    --lr $Lr `
    --l2 $L2 `
    --optimizer $Optimizer `
    --dropout 0.1 `
    --prompt-type $PromptType `
    --mask-type "constant-$MaskRate" `
    --lambda-proto $LambdaProto `
    --lambda-rel $LambdaRel `
    --seed $Seed
