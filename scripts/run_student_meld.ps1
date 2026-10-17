# Needs a MELD teacher checkpoint first, trained with the same -RobertaIdx.

param(
    [double]$MaskRate = 0.1,
    [string]$PromptType = "none",
    [double]$LambdaProb = 0.4,
    [double]$LambdaProto = 1.0,
    [double]$LambdaRel = 0.8,
    [double]$Lr = 0.00005,
    [double]$L2 = 0.0001,
    [int]$Seed = 50,
    [int]$RobertaIdx = 1,
    [string]$Optimizer = 'adamw'
)

Set-Location -Path (Join-Path $PSScriptRoot '..')

& ".\fsce_net\Scripts\python.exe" train/train_student_meld.py `
    --lstm_hidden_size 128 `
    --graph-layers 1 `
    --hg-layers 1 `
    --lr $Lr `
    --optimizer $Optimizer `
    --l2 $L2 `
    --dropout 0.1 `
    --prompt-type $PromptType `
    --mask-type "constant-$MaskRate" `
    --lambda-prob $LambdaProb `
    --lambda-proto $LambdaProto `
    --lambda-rel $LambdaRel `
    --roberta-idx $RobertaIdx `
    --seed $Seed
