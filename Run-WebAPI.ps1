<#
.SYNOPSIS
    Launches the IWFM FastAPI Web Viewer application.

.DESCRIPTION
    This script sets up the Python environment and launches the IWFM FastAPI
    web visualization application via the pyiwfm CLI (viewer subcommand).

    The FastAPI viewer uses client-side vtk.js rendering with a React frontend,
    deck.gl map visualization, and Plotly budget charts. It provides better
    performance than the legacy Trame-based viewer for large models.

    The script will:
    1. Check for Python 3.10+ installation
    2. Create/activate a virtual environment (optional)
    3. Install required dependencies if missing
    4. Build the React frontend if needed (optional)
    5. Launch the FastAPI web viewer application

.PARAMETER ModelDir
    Path to the IWFM model directory. Default: ~/OneDrive/Desktop/c2vsimfg

.PARAMETER BindAddress
    Server host/bind address. Default: 127.0.0.1
    Use 0.0.0.0 to allow connections from other machines on the network.

.PARAMETER Port
    HTTP port for the web server. Default: 8080

.PARAMETER NoBrowser
    Don't automatically open the browser.

.PARAMETER UseVenv
    Create and use a Python virtual environment.

.PARAMETER InstallDeps
    Install/update dependencies before running.

.PARAMETER BuildFrontend
    Build the React frontend before running (requires Node.js/npm).

.PARAMETER Preprocessor
    Load from preprocessor file only (faster, mesh-only visualization).

.PARAMETER Simulation
    Load from simulation file (full model with all components).

.PARAMETER Title
    Custom application title. Default: auto-detected from model name.

.PARAMETER CRS
    Source coordinate reference system string for reprojection.
    Default: "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
    (UTM Zone 10N, NAD83, US survey feet - appropriate for C2VSimFG).

.PARAMETER DebugMode
    Enable debug mode with verbose logging and Swagger API docs at /api/docs.

.PARAMETER Help
    Show this help message.

.EXAMPLE
    .\Run-WebAPI.ps1
    Runs the FastAPI viewer with default settings (C2VSimFG from Desktop).

.EXAMPLE
    .\Run-WebAPI.ps1 -ModelDir "C:\Models\C2VSimFG" -Port 9000
    Runs the viewer for a specific model on port 9000.

.EXAMPLE
    .\Run-WebAPI.ps1 -BindAddress 0.0.0.0 -Port 8080
    Runs the viewer accessible from other machines on the network.

.EXAMPLE
    .\Run-WebAPI.ps1 -UseVenv -InstallDeps
    Creates a virtual environment and installs dependencies first.

.EXAMPLE
    .\Run-WebAPI.ps1 -Preprocessor "Preprocessor\C2VSimFG_Preprocessor.in"
    Loads only the preprocessor file for faster mesh-only visualization.

.EXAMPLE
    .\Run-WebAPI.ps1 -DebugMode
    Runs the viewer with debug logging and Swagger API docs enabled.

.EXAMPLE
    .\Run-WebAPI.ps1 -CRS "EPSG:26910"
    Runs with a specific coordinate reference system.

.EXAMPLE
    .\Run-WebAPI.ps1 -Help
    Shows this help message.

.NOTES
    Author: California Department of Water Resources
    License: GPL-2.0

    This viewer requires the [webapi] extras:
        pip install pyiwfm[webapi]

    Dependencies: fastapi, uvicorn, pydantic, pyvista, vtk, pyproj
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$ModelDir = "",

    [string]$BindAddress = "127.0.0.1",

    [int]$Port = 8080,

    [switch]$NoBrowser,

    [switch]$UseVenv,

    [switch]$InstallDeps,

    [switch]$BuildFrontend,

    [string]$Preprocessor = "",

    [string]$Simulation = "",

    [string]$Title = "",

    [string]$CRS = "",

    [switch]$DebugMode,

    [switch]$Help
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── Help ──────────────────────────────────────────────────────────────
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# ── Banner ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  =========================================================" -ForegroundColor Cyan
Write-Host "         IWFM FastAPI Web Viewer" -ForegroundColor Cyan
Write-Host "  =========================================================" -ForegroundColor Cyan
Write-Host ""

# ── Resolve model directory ──────────────────────────────────────────
if ([string]::IsNullOrEmpty($ModelDir)) {
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
    $ModelDir = Join-Path $DesktopPath "c2vsimfg"

    # Check OneDrive Desktop fallback
    $OneDriveDesktop = Join-Path $env:USERPROFILE "OneDrive\Desktop\c2vsimfg"
    if ((Test-Path $OneDriveDesktop) -and -not (Test-Path $ModelDir)) {
        $ModelDir = $OneDriveDesktop
    }
}

$ModelDir = [System.IO.Path]::GetFullPath($ModelDir)

Write-Host "  Model Directory : $ModelDir" -ForegroundColor White
Write-Host "  Server Address  : http://${BindAddress}:${Port}" -ForegroundColor White
if ($DebugMode) {
    Write-Host "  API Docs        : http://${BindAddress}:${Port}/api/docs" -ForegroundColor White
}
Write-Host ""

# ── Find Python 3.10+ ────────────────────────────────────────────────
function Find-Python {
    $pythonCommands = @("python", "python3", "py -3")

    foreach ($cmd in $pythonCommands) {
        try {
            $cmdParts = $cmd -split " "
            $exe = $cmdParts[0]
            $extraArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length - 1)] } else { @() }

            $version = & $exe @extraArgs --version 2>&1
            if ($version -match "Python 3\.(\d+)") {
                $minor = [int]$Matches[1]
                if ($minor -ge 10) {
                    Write-Host "  Python          : $version" -ForegroundColor Green
                    return $cmd
                }
            }
        }
        catch {
            # Try next command
        }
    }

    return $null
}

$Python = Find-Python
if (-not $Python) {
    Write-Host "ERROR: Python 3.10+ not found." -ForegroundColor Red
    Write-Host "Please install Python 3.10 or later from https://python.org" -ForegroundColor Yellow
    exit 1
}

# ── Virtual environment setup ─────────────────────────────────────────
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if ($UseVenv) {
    if (-not (Test-Path $VenvDir)) {
        Write-Host ""
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        $pythonParts = $Python -split " "
        & $pythonParts[0] @($pythonParts[1..($pythonParts.Length - 1)] + @("-m", "venv", $VenvDir))

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
            exit 1
        }
        Write-Host "Virtual environment created at: $VenvDir" -ForegroundColor Green
        $InstallDeps = $true
    }

    if (Test-Path $VenvPython) {
        $Python = $VenvPython
        Write-Host "  Venv            : $VenvDir" -ForegroundColor Green
    }
    else {
        Write-Host "ERROR: Virtual environment Python not found" -ForegroundColor Red
        exit 1
    }
}

# ── Install dependencies ──────────────────────────────────────────────
if ($InstallDeps) {
    Write-Host ""
    Write-Host "Installing dependencies..." -ForegroundColor Yellow

    # Check if pyiwfm source is available for editable install
    $PyiwfmSetup = Join-Path $ScriptDir "pyproject.toml"

    if (Test-Path $PyiwfmSetup) {
        Write-Host "  Installing pyiwfm with [webapi] extras..." -ForegroundColor Gray
        & $Python -m pip install --upgrade pip --quiet
        & $Python -m pip install -e "$ScriptDir[webapi]"

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to install pyiwfm[webapi]" -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "  Installing FastAPI web viewer dependencies..." -ForegroundColor Gray
        $deps = @(
            "numpy>=1.21",
            "pandas>=1.3",
            "h5py>=3.0",
            "jinja2>=3.0",
            "pyvista>=0.43",
            "vtk>=9.2",
            "fastapi>=0.100",
            "uvicorn>=0.23",
            "pydantic>=2.0",
            "pyproj>=3.4",
            "python-multipart>=0.0.6"
        )

        foreach ($dep in $deps) {
            Write-Host "    $dep" -ForegroundColor Gray
            & $Python -m pip install $dep --quiet
        }
    }

    Write-Host "Dependencies installed." -ForegroundColor Green
    Write-Host ""
}

# ── Build frontend (optional) ─────────────────────────────────────────
if ($BuildFrontend) {
    $FrontendDir = Join-Path $ScriptDir "frontend"

    if (-not (Test-Path $FrontendDir)) {
        Write-Host "WARNING: frontend/ directory not found - skipping build." -ForegroundColor Yellow
    }
    else {
        Write-Host "Building React frontend..." -ForegroundColor Yellow

        $NpmCmd = Get-Command npm -ErrorAction SilentlyContinue
        if (-not $NpmCmd) {
            Write-Host "ERROR: npm not found. Install Node.js from https://nodejs.org" -ForegroundColor Red
            exit 1
        }

        Push-Location $FrontendDir
        try {
            if (-not (Test-Path "node_modules")) {
                Write-Host "  npm install..." -ForegroundColor Gray
                & npm install
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "ERROR: npm install failed" -ForegroundColor Red
                    exit 1
                }
            }

            Write-Host "  npm run build..." -ForegroundColor Gray
            & npm run build
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Frontend build failed" -ForegroundColor Red
                exit 1
            }
            Write-Host "Frontend built successfully." -ForegroundColor Green
        }
        finally {
            Pop-Location
        }
        Write-Host ""
    }
}

# ── Check model directory ─────────────────────────────────────────────
if (-not (Test-Path $ModelDir)) {
    Write-Host "WARNING: Model directory not found: $ModelDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can download C2VSimFG from:" -ForegroundColor White
    Write-Host "  https://data.cnra.ca.gov/dataset/c2vsimfg" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or specify a model directory:" -ForegroundColor White
    Write-Host "  .\Run-WebAPI.ps1 -ModelDir ""C:\path\to\model""" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# ── Build CLI arguments ───────────────────────────────────────────────
$CliArgs = @(
    "-m", "pyiwfm", "viewer",
    "--model-dir", $ModelDir,
    "--host", $BindAddress,
    "--port", $Port
)

if (-not [string]::IsNullOrEmpty($Title)) {
    $CliArgs += @("--title", $Title)
}

if ($NoBrowser) {
    $CliArgs += "--no-browser"
}

if (-not [string]::IsNullOrEmpty($Preprocessor)) {
    $CliArgs += @("--preprocessor", $Preprocessor)
}

if (-not [string]::IsNullOrEmpty($Simulation)) {
    $CliArgs += @("--simulation", $Simulation)
}

if (-not [string]::IsNullOrEmpty($CRS)) {
    $CliArgs += @("--crs", $CRS)
}

if ($DebugMode) {
    $CliArgs += "--debug"
}

# ── Launch ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Launching IWFM FastAPI Web Viewer..." -ForegroundColor Cyan
Write-Host "  URL: http://${BindAddress}:${Port}" -ForegroundColor White
if ($DebugMode) {
    Write-Host "  API Docs: http://${BindAddress}:${Port}/api/docs" -ForegroundColor White
}
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

try {
    & $Python @CliArgs
}
catch {
    if ($_.Exception.Message -notmatch "keyboard interrupt") {
        Write-Host ""
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        Write-Host "Troubleshooting:" -ForegroundColor Yellow
        Write-Host "  1. Install deps:  .\Run-WebAPI.ps1 -InstallDeps" -ForegroundColor Gray
        Write-Host "  2. Build frontend: .\Run-WebAPI.ps1 -BuildFrontend" -ForegroundColor Gray
        Write-Host "  3. Enable debug:  .\Run-WebAPI.ps1 -DebugMode" -ForegroundColor Gray
        exit 1
    }
}

Write-Host ""
Write-Host "Viewer stopped." -ForegroundColor Green
