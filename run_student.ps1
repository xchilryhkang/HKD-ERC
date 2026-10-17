# Runs train_student.py — distills the student (incomplete modality,
# prompt_type='prototype', mask_rate=0.1 by default) from the teacher
# checkpoints produced by run_teacher.ps1 / train_teacher.py.
#
# Architecture flags here MUST match whatever run_teacher.ps1 used (see
# train_student.py's module docstring) — kept in sync with that script's
# tuned values. --lr/--l2/--dropout/--prompt-type/--mask-type are student-
# only and free to differ from the teacher's own training run.
#
# Usage (PowerShell): .\run_student.ps1 [-Dataset <name>] [-MaskRate <rate>] [-PromptType <type>]
#                      [-LambdaProto <w>] [-LambdaRel <w>]
# Dataset defaults to IEMOCAPSix, MaskRate defaults to 0.1, PromptType
# defaults to "prototype" (choices: none, constant, prototype). Distillation
# loss is 4 clusters (CE, prob, proto, rel) -- see train_student.py's module
# docstring. CE and prob are fixed at weight 1; only LambdaProto/LambdaRel
# are tunable here (each defaults to 1.0, set to 0 to ablate that cluster).
# LambdaRel covers the 'e' (VUE/consensus) node's own relational-prototype
# loss plus its structural relation to a/t/v after hypergraph -- see
# kd_losses.py's PrototypeRelationLoss / NodeRelationLoss docstrings.
# Requires teacher checkpoints for that dataset to already exist under
# output/model/teacher/ — run run_teacher.ps1 first if they don't.

param(
    [string]$Dataset = "IEMOCAPSix",
    [double]$MaskRate = 0.1,
    [string]$PromptType = "prototype",
    [double]$LambdaProto = 1.0,
    [double]$LambdaRel = 1.0
)

Set-Location -Path $PSScriptRoot

& ".\fsce_net\Scripts\python.exe" train_student.py `
    --dataset $Dataset `
    --lstm_hidden_size 128 `
    --graph-layers 1 `
    --hg-layers 1 `
    --lr 0.0001 `
    --l2 0.00001 `
    --dropout 0.1 `
    --prompt-type $PromptType `
    --mask-type "constant-$MaskRate" `
    --lambda-proto $LambdaProto `
    --lambda-rel $LambdaRel
