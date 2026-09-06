<#
.SYNOPSIS
    Runs every math-verify arm once, in order, as a smoke test of the harness.

.DESCRIPTION
    The `math-verify-*` files in configs/ are the single-task versions of the
    six experiment arms: the two baselines, the three solutions, and
    Self-Collaboration as published. Running all of them end to end exercises
    every part a real experiment depends on -- configuration validation, the
    benchmark pull, Docker, the model backend, the tester sidecar, and the
    results written to jobs/ -- against one task rather than the whole
    benchmark.

    This is not free. The arms keep the `n_attempts` their configurations
    state, so the six runs are twelve trials in total (four each for
    Single-Shot and CodeTeam, one each for the rest) against the real model
    backend.

    Each arm runs as its own job under jobs/, sharing a `smoke__<timestamp>`
    prefix so the six read as one sitting. A failing arm does not stop the
    others: the point of the run is to learn which parts work, so failures are
    collected and reported together at the end. Pass -StopOnFailure for the
    opposite. The exit code is 0 only when every arm succeeded.

.EXAMPLE
    .\scripts\run-smoke-tests.ps1

.EXAMPLE
    .\scripts\run-smoke-tests.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [switch]$StopOnFailure,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# Baselines first. They are the simplest arms, so a harness that is broken for
# every arm says so in the first minutes rather than the last.
$Arms = @(
    'math-verify-single-shot'
    'math-verify-terminus'
    'math-verify-self-collaboration'
#     'math-verify-self-collaboration-as-published'
    'math-verify-codeteam'
#     'math-verify-codes'
)

$Root   = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Stamp  = Get-Date -Format 'yyyy-MM-dd__HH-mm-ss'

if (-not (Test-Path $Python)) {
    throw "No interpreter at $Python. Create the virtual environment first."
}
if (-not (Test-Path (Join-Path $Root '.env'))) {
    throw "No .env in $Root. Copy .env.example and set API_KEY."
}

# Resolve every configuration up front: a missing file in the last arm should
# not be discovered an hour into the run.
$Configs = foreach ($arm in $Arms) {
    $path = Join-Path $Root "configs\$arm.yaml"
    if (-not (Test-Path $path)) { throw "Missing configuration: $path" }
    $path
}

Write-Host "Smoke run $Stamp -- $($Arms.Count) arms, one task each" -ForegroundColor Cyan

$Results = @()
Push-Location $Root
try {
    for ($i = 0; $i -lt $Arms.Count; $i++) {
        $arm     = $Arms[$i]
        $jobName = "smoke__${Stamp}__$arm"

        Write-Host ''
        Write-Host "[$($i + 1)/$($Arms.Count)] $arm -> jobs\$jobName" -ForegroundColor Cyan

        if ($DryRun) {
            Write-Host "  $Python main.py --config configs\$arm.yaml --job-name $jobName"
            continue
        }

        $started = Get-Date
        & $Python (Join-Path $Root 'main.py') --config $Configs[$i] --job-name $jobName
        $code    = $LASTEXITCODE
        $elapsed = (Get-Date) - $started

        $Results += [pscustomobject]@{
            Arm     = $arm
            Exit    = $code
            Minutes = [math]::Round($elapsed.TotalMinutes, 1)
            Job     = "jobs\$jobName"
        }

        if ($code -ne 0) {
            Write-Host "  FAILED (exit $code)" -ForegroundColor Red
            if ($StopOnFailure) { break }
        }
    }
}
finally {
    Pop-Location
}

if ($DryRun) { return }

Write-Host ''
$Results | Format-Table -AutoSize

$failed = @($Results | Where-Object { $_.Exit -ne 0 })
if ($failed.Count -gt 0) {
    Write-Host "$($failed.Count) of $($Results.Count) arms failed: $($failed.Arm -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "All $($Results.Count) arms completed." -ForegroundColor Green
exit 0
