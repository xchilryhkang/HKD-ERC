param(
    [ValidateSet('CMUMOSI', 'CMUMOSEI')]
    [string]$Dataset = 'CMUMOSI',
    [int]$Seed = 50,
    [double]$Lr = 0.00005,
    [double]$L2 = 0.0001,
    [string]$Optimizer = 'adamw'
)

Set-Location -Path (Join-Path $PSScriptRoot '..')

& ".\fsce_net\Scripts\python.exe" train/train_teacher_cmu.py `
    --dataset $Dataset `
    --lstm_hidden_size 128 `
    --lr $Lr `
    --optimizer $Optimizer `
    --l2 $L2 `
    --graph-layers 1 `
    --hg-layers 1 `
    --dropout 0.1 `
    --seed $Seed
