# Compile best.pt -> best.hef for the Hailo-8, from Windows via WSL2 -- no Docker.
#
# hailo-compile/hailo-compile.ps1 shells into Docker Desktop because that's the only
# way to get "Linux" on macOS. Windows already has real Linux via WSL2, so
# this launches hailo-compile/hailo-compile-wsl.sh directly inside the default WSL
# distro instead -- one less virtualization layer, and the Windows-side
# NVIDIA driver passes through to it for HEF_GPU=1.
#
# WSL auto-translates the invoking PowerShell session's cwd, so run this from
# the repo root the same way as the Docker script -- no -v/path mounting.
#
# Usage:
#   $env:HEF_RUN = "poc-v2-480"; $env:HEF_GPU = "1"
#   .\docker\hailo-compile-wsl.ps1
#
# Requires WSL2 with Ubuntu 22.04 (`wsl --install -d Ubuntu-22.04`) and the
# DFC wheel reachable from inside it -- see hailo-compile/hailo-compile-wsl.sh for
# where HEF_WHEEL is looked for by default.
$ErrorActionPreference = "Stop"

$envArgs = @()
foreach ($name in "HEF_RUN", "HEF_DATASET", "HEF_IMGSZ", "HEF_CONF", "HEF_FRACTION", "HEF_FORCE", "HEF_A16_CLS", "HEF_GPU", "HEF_WHEEL", "HEF_WHEEL_URL", "HEF_WHEEL_SHA256", "HEF_VENV") {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($value) { $envArgs += "$name=$value" }
}

wsl.exe -e env @envArgs bash hailo-compile/hailo-compile-wsl.sh
