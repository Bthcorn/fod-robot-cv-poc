# Compile best.pt -> best.hef for the Hailo-8, from Windows (Docker Desktop).
#
# Same payload as the Mac driver -- hailo-compile/hailo-compile.sh runs unchanged inside
# the container, it doesn't care what host started it. This script only replaces
# the macOS/zsh docker-run invocation documented in README.md with a PowerShell
# equivalent: $PWD/$HOME -> $PWD.Path/$env:USERPROFILE, and `bash -s < file`
# (PowerShell has no `<` redirection into external processes) -> `Get-Content | ...`.
#
# On Windows x86_64 there's no Rosetta/AVX concern -- the amd64 container runs
# natively, no emulation -- but the >=10 GB VM RAM requirement still applies:
# Docker Desktop -> Settings -> Resources (WSL2 backend: edit %UserProfile%\.wslconfig
# instead if Resources is greyed out).
#
# Usage:
#   $env:HEF_RUN = "poc-v2-480"; $env:HEF_DATASET = "fod-a-3k"; $env:HEF_IMGSZ = "480"; $env:HEF_CONF = "0.10"
#   .\docker\hailo-compile.ps1
#
# Unset variables are left for hailo-compile/hailo-compile.sh's own ${VAR:-default} fallbacks.
$ErrorActionPreference = "Stop"

$envArgs = @()
foreach ($name in "HEF_RUN", "HEF_DATASET", "HEF_IMGSZ", "HEF_CONF") {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($value) { $envArgs += "-e"; $envArgs += "$name=$value" }
}

Get-Content "$PSScriptRoot\hailo-compile.sh" -Raw | docker --context desktop-linux run --platform linux/amd64 --rm -i `
    -v "$($PWD.Path):/work" -w /work `
    -v "$env:USERPROFILE\Downloads:/wheels:ro" `
    -v hailo-pipcache:/root/.cache/pip `
    @envArgs `
    python:3.10-slim bash -s
