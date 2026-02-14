<#
.SYNOPSIS
    Run IWFM roundtrip test models (preprocessor and/or simulation).

.DESCRIPTION
    Runs the IWFM PreProcessor and/or Simulation for the text and/or DSS
    versions of the C2VSimFG model written by setup_roundtrip.py.

    Each executable prompts for the main input file name via stdin.
    This script pipes the correct filename to each executable.

.PARAMETER Model
    Which model variant to run: "text", "dss", or "both" (default: "both").

.PARAMETER RunPreprocessor
    Run the PreProcessor step.

.PARAMETER RunSimulation
    Run the Simulation step.

.PARAMETER RunAll
    Shortcut for -RunPreprocessor -RunSimulation.

.PARAMETER BaseDir
    Base output directory (default: C:\temp\iwfm_roundtrip).

.EXAMPLE
    .\run_roundtrip.ps1 -Model text -RunAll
    .\run_roundtrip.ps1 -Model dss -RunPreprocessor
    .\run_roundtrip.ps1 -Model both -RunSimulation
    .\run_roundtrip.ps1 -RunAll
#>

[CmdletBinding()]
param(
    [ValidateSet("text", "dss", "both")]
    [string]$Model = "both",

    [switch]$RunPreprocessor,
    [switch]$RunSimulation,
    [switch]$RunAll,

    [string]$BaseDir = "C:\temp\iwfm_roundtrip"
)

# If -RunAll, enable both steps
if ($RunAll) {
    $RunPreprocessor = $true
    $RunSimulation = $true
}

if (-not $RunPreprocessor -and -not $RunSimulation) {
    Write-Host "ERROR: Specify at least one of: -RunPreprocessor, -RunSimulation, or -RunAll" -ForegroundColor Red
    exit 1
}

# Build list of model variants to run
$variants = @()
if ($Model -eq "both") {
    $variants = @("text", "dss")
} else {
    $variants = @($Model)
}

# Validate directories exist
foreach ($variant in $variants) {
    $dir = Join-Path $BaseDir $variant
    if (-not (Test-Path $dir)) {
        Write-Host "ERROR: Model directory not found: $dir" -ForegroundColor Red
        Write-Host "Run setup_roundtrip.py first to create the model files." -ForegroundColor Yellow
        exit 1
    }
}

function Run-IWFMExecutable {
    param(
        [string]$ExeName,
        [string]$WorkingDir,
        [string]$InputFileName,
        [string]$Label
    )

    $exePath = Join-Path (Split-Path $WorkingDir -Parent) $ExeName
    if (-not (Test-Path $exePath)) {
        # Also check in working dir itself
        $exePath = Join-Path $WorkingDir $ExeName
    }
    if (-not (Test-Path $exePath)) {
        Write-Host "  ERROR: $ExeName not found at expected location" -ForegroundColor Red
        return $false
    }

    $logFile = Join-Path $WorkingDir "${Label}_log.txt"

    Write-Host "  Executable: $exePath"
    Write-Host "  Working dir: $WorkingDir"
    Write-Host "  Input file: $InputFileName"
    Write-Host "  Log file: $logFile"
    Write-Host ""

    $startTime = Get-Date

    # Pipe the input file name to the executable
    # Use Start-Process with redirected stdin via cmd /c echo
    $proc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "echo $InputFileName | `"$exePath`" > `"$logFile`" 2>&1" `
        -WorkingDirectory $WorkingDir `
        -NoNewWindow `
        -Wait `
        -PassThru

    $endTime = Get-Date
    $elapsed = $endTime - $startTime

    $exitCode = $proc.ExitCode
    $hours = [math]::Floor($elapsed.TotalHours)
    $minutes = $elapsed.Minutes
    $seconds = $elapsed.Seconds

    if ($exitCode -eq 0) {
        Write-Host "  COMPLETED: exit code 0 (${hours}h ${minutes}m ${seconds}s)" -ForegroundColor Green
    } else {
        Write-Host "  FAILED: exit code $exitCode (${hours}h ${minutes}m ${seconds}s)" -ForegroundColor Red
    }

    return ($exitCode -eq 0)
}


# ================================================================
# Main execution
# ================================================================
$overallStart = Get-Date
$results = @{}

foreach ($variant in $variants) {
    $modelDir = Join-Path $BaseDir $variant
    $ppDir = Join-Path $modelDir "Preprocessor"
    $simDir = Join-Path $modelDir "Simulation"

    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Model variant: $variant" -ForegroundColor Cyan
    Write-Host "  Directory: $modelDir" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    # --- PreProcessor ---
    if ($RunPreprocessor) {
        Write-Host "--- PreProcessor ($variant) ---" -ForegroundColor Yellow

        if (-not (Test-Path $ppDir)) {
            Write-Host "  ERROR: Preprocessor directory not found: $ppDir" -ForegroundColor Red
            $results["${variant}_preprocessor"] = "SKIPPED"
        } else {
            $success = Run-IWFMExecutable `
                -ExeName "PreProcessor_x64.exe" `
                -WorkingDir $ppDir `
                -InputFileName "Preprocessor.in" `
                -Label "preprocessor"

            $results["${variant}_preprocessor"] = if ($success) { "OK" } else { "FAILED" }
        }
        Write-Host ""
    }

    # --- Simulation ---
    if ($RunSimulation) {
        Write-Host "--- Simulation ($variant) ---" -ForegroundColor Yellow

        if (-not (Test-Path $simDir)) {
            Write-Host "  ERROR: Simulation directory not found: $simDir" -ForegroundColor Red
            $results["${variant}_simulation"] = "SKIPPED"
        } else {
            $success = Run-IWFMExecutable `
                -ExeName "Simulation_x64.exe" `
                -WorkingDir $simDir `
                -InputFileName "Simulation_MAIN.IN" `
                -Label "simulation"

            $results["${variant}_simulation"] = if ($success) { "OK" } else { "FAILED" }
        }
        Write-Host ""
    }
}

# ================================================================
# Summary
# ================================================================
$overallEnd = Get-Date
$overallElapsed = $overallEnd - $overallStart
$oh = [math]::Floor($overallElapsed.TotalHours)
$om = $overallElapsed.Minutes
$os = $overallElapsed.Seconds

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Total elapsed: ${oh}h ${om}m ${os}s"
Write-Host ""

foreach ($key in $results.Keys | Sort-Object) {
    $status = $results[$key]
    $color = if ($status -eq "OK") { "Green" } elseif ($status -eq "FAILED") { "Red" } else { "Yellow" }
    Write-Host "  $key : $status" -ForegroundColor $color
}

Write-Host ""
Write-Host "Log files:"
foreach ($variant in $variants) {
    $modelDir = Join-Path $BaseDir $variant
    if ($RunPreprocessor) {
        $log = Join-Path $modelDir "Preprocessor\preprocessor_log.txt"
        if (Test-Path $log) { Write-Host "  $log" }
    }
    if ($RunSimulation) {
        $log = Join-Path $modelDir "Simulation\simulation_log.txt"
        if (Test-Path $log) { Write-Host "  $log" }
    }
}
