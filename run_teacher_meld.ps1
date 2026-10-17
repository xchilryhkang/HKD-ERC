# Runs train_teacher_meld.py -- MELD has a genuine train(1038)/valid(114)/
# test(280) split (see dataloader_meld.MELDRobertaDataset's docstring), so
# early stopping tracks the REAL valid split every epoch, unlike the
# IEMOCAP scripts which track the held-out session directly (no separate
# val exists there).
#
# Usage (PowerShell): .\run_teacher_meld.ps1 [-Seed <n>] [-Lr <lr>] [-L2 <l2>] [-RobertaIdx <1-4>]
# Seed defaults to 50. Lr/L2 default to 0.00005/0.0001 (tuned via
# tune_lr_l2_meld.py -- see output/tune/*_lr_l2_sweep_meld.txt). RobertaIdx
# picks which of the 4 RoBERTa text variants to use, defaults to 1.

param(
    [int]$Seed = 50,
    [double]$Lr = 0.00005,
    [double]$L2 = 0.0001,
    [int]$RobertaIdx = 1
)

Set-Location -Path $PSScriptRoot

& ".\fsce_net\Scripts\python.exe" train_teacher_meld.py `
    --lstm_hidden_size 128 `
    --lr $Lr `
    --l2 $L2 `
    --graph-layers 1 `
    --hg-layers 1 `
    --dropout 0.1 `
    --roberta-idx $RobertaIdx `
    --seed $Seed
