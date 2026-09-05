# Compile best.pt -> best.hef for the Hailo-8, from Windows via WSL2.
#
# The compiler is x86_64-Linux-only, and WSL2 is real Linux -- a kernel, not
# emulation -- so this just launches hailo-compile/hailo-compile.sh inside the
# default WSL distro. The Windows-side NVIDIA driver passes through to it, which
# is what HEF_GPU=1 needs.
#
# WSL auto-translates the invoking PowerShell session's cwd, so run this from the
# repo root -- nothing to mount.
#
# Usage:
#   $env:HEF_RUN = "arg-bolts-4-n-640"; $env:HEF_GPU = "1"
#   .\hailo-compile\hailo-compile.ps1
#
# Requires WSL2 with Ubuntu 22.04 (`wsl --install -d Ubuntu-22.04`) and the DFC
# wheel reachable from inside it -- see hailo-compile/hailo-compile.sh for where
# HEF_WHEEL is looked for by default.
$ErrorActionPreference = "Stop"

$envArgs = @()
foreach ($name in "HEF_RUN", "HEF_DATASET", "HEF_IMGSZ", "HEF_CONF", "HEF_FRACTION", "HEF_FORCE", "HEF_A16_CLS", "HEF_GPU", "HEF_WHEEL", "HEF_WHEEL_URL", "HEF_WHEEL_SHA256", "HEF_VENV") {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($value) { $envArgs += "$name=$value" }
}

wsl.exe -e env @envArgs bash hailo-compile/hailo-compile.sh
