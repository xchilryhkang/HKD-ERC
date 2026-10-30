# Needs a CMU teacher checkpoint first, trained on the same -Dataset.

param(
    [ValidateSet('CMUMOSI', 'CMUMOSEI')]
    [string]$Dataset = 'CMUMOSI',
    [double]$MaskRate = 0.1,
    [string]$PromptType = 'none',
    [double]$LambdaProb = 1.0,
    [double]$LambdaProto = 1.0,
    [double]$LambdaRel = 1.0,
    [string]$LossNorm = 'none',
    [double]$Lr = 0.00005,
    [double]$L2 = 0.0001,
    [int]$Seed = 50,
    [string]$Optimizer = 'adamw'
)

Set-Location -Path (Join-Path $PSScriptRoot '..')

& ".\fsce_net\Scripts\python.exe" train/train_student_cmu.py `
    --dataset $Dataset `
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
    --loss-norm $LossNorm `
    --seed $Seed
