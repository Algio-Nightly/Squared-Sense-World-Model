<#
.SYNOPSIS
    Launches GPU training scripts seamlessly inside WSL2 with CUDA acceleration from Windows PowerShell.

.DESCRIPTION
    Bridges execution from Windows into Debian WSL2 using the dedicated .venv-wsl environment,
    with automatic path mapping, VRAM safety caps, and zero manual terminal switching.

.EXAMPLE
    .\run_gpu.ps1 train/train_spatial_jepa.py --data_dir data/episodes --epochs 80
#>

$ErrorActionPreference = "Stop"
$ScriptArgs = $args

$repoRootWin = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check WSL Debian availability
& wsl.exe -d Debian -e true 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "[run_gpu.ps1] Debian distribution not accessible in WSL. Run 'wsl -l -v' to check installed distros."
    exit 1
}

# Translate Windows path to WSL /mnt/... path
$driveLetter = $repoRootWin.Substring(0, 1).ToLower()
$pathWithoutDrive = $repoRootWin.Substring(2).Replace('\', '/')
$repoRootWsl = "/mnt/$driveLetter$pathWithoutDrive"

# Format arguments for bash
$joinedArgs = ($ScriptArgs | ForEach-Object {
    if ($_ -match '\s') { "'$_'" } else { $_ }
}) -join " "

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Squared Sense World Model — Windows -> WSL2 GPU Bridge" -ForegroundColor Cyan
Write-Host "Windows Root : $repoRootWin"
Write-Host "WSL2 Root    : $repoRootWsl"
Write-Host "Executing    : ./run_gpu.sh $joinedArgs"
Write-Host "==================================================================" -ForegroundColor Cyan

# Execute inside Debian WSL with Windows directory passed to --cd
& wsl.exe -d Debian --cd $repoRootWin bash ./run_gpu.sh @ScriptArgs
exit $LASTEXITCODE

