# Needs a teacher checkpoint under output/model/teacher/ first.

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
