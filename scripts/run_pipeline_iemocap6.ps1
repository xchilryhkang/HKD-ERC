param(
    [double]$MaskRate = 0.1,
    [string]$PromptType = "prototype",
    [int]$Seed = 50
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot '..')

Write-Host "`n===== [1/3] Training teacher (IEMOCAPSix, 5-fold leave-one-session-out) =====" -ForegroundColor Cyan
& "$PSScriptRoot\run_teacher_iemocap6.ps1" -Seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Teacher training failed (exit $LASTEXITCODE)" }

Write-Host "`n===== [2/3] Training student (IEMOCAPSix, 5-fold leave-one-session-out) =====" -ForegroundColor Cyan
& "$PSScriptRoot\run_student_iemocap6.ps1" -MaskRate $MaskRate -PromptType $PromptType -Seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Student training failed (exit $LASTEXITCODE)" }

Write-Host "`n===== [3/3] Running analysis (metrics + t-SNE, all 5 folds) =====" -ForegroundColor Cyan
& ".\fsce_net\Scripts\python.exe" analysis/analyze_distribution.py `
    --dataset IEMOCAPSix --folds 0,1,2,3,4 `
    --mask-rate $MaskRate --prompt-type $PromptType
if ($LASTEXITCODE -ne 0) { throw "Analysis failed (exit $LASTEXITCODE)" }

Write-Host "`n===== Pipeline complete =====" -ForegroundColor Green
