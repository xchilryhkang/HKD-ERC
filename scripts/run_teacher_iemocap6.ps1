# Runs train_teacher_iemocap.py for IEMOCAPSix -- 5-fold leave-one-session-
# out CV, best checkpoint = best session-out F1 (early stopping tracks the
# held-out session directly every epoch; see train_teacher_iemocap.py's
# module docstring for why there's no separate validation session under
# this protocol).
#
# Usage (PowerShell): .\run_teacher_iemocap6.ps1 [-Seed <n>] [-Lr <lr>] [-L2 <l2>] [-Optimizer <adam|adamw>]
# Seed defaults to 50. Lr/L2/Optimizer default to 0.0002/0.00001/adamw (from tune_lr_l2_iemocap.py).

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
