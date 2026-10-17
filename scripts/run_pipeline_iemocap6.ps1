# Runs the full IEMOCAPSix pipeline end-to-end, unattended:
#   [1/3] train teacher   (run_teacher_iemocap6.ps1 -- 5-fold leave-one-session-out CV)
#   [2/3] train student    (run_student_iemocap6.ps1 -- same 5 folds, distills from the teacher just trained)
#   [3/3] analysis          (analyze_distribution.py -- t-SNE/metrics report over all 5 folds)
#
# Stops immediately if any step fails (non-zero exit code) rather than
# silently continuing to the next step on bad state.
#
# Usage (PowerShell): .\run_pipeline_iemocap6.ps1 [-MaskRate <rate>] [-PromptType <type>] [-Seed <n>]
# MaskRate/PromptType are forwarded to both the student training step and
# the analysis step (so the analysis uses the SAME mask_rate/prompt_type
# the student was actually trained+evaluated with). Defaults: 0.1, prototype.
# Seed is forwarded to both the teacher and student training steps (default 50).
#
# Teacher-side hyperparameters (lr/l2/dropout/optimizer/...) and student-side
# distillation lambdas all stay at run_teacher_iemocap6.ps1 /
# run_student_iemocap6.ps1's own defaults -- edit those files directly if
# you need to change them, rather than piling on more flags here.

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
