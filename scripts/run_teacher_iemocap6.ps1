param(
    [int]$Seed = 50,
    [double]$Lr = 0.0002,
    [double]$L2 = 0.00001,
    [string]$Optimizer = 'adamw'
)

Set-Location -Path (Join-Path $PSScriptRoot '..')

& ".\fsce_net\Scripts\python.exe" train/train_teacher_iemocap.py `
    --dataset IEMOCAPSix `
    --lstm_hidden_size 128 `
    --lr $Lr `
    --l2 $L2 `
    --optimizer $Optimizer `
    --graph-layers 1 `
    --hg-layers 1 `
    --dropout 0.1 `
    --seed $Seed
