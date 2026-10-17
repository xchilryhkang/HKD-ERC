param(
    [int]$Seed = 50,
    [double]$Lr = 0.00005,
    [double]$L2 = 0.0001,
    [int]$RobertaIdx = 1
)

Set-Location -Path (Join-Path $PSScriptRoot '..')

& ".\fsce_net\Scripts\python.exe" train/train_teacher_meld.py `
    --lstm_hidden_size 128 `
    --lr $Lr `
    --l2 $L2 `
    --graph-layers 1 `
    --hg-layers 1 `
    --dropout 0.1 `
    --roberta-idx $RobertaIdx `
    --seed $Seed
