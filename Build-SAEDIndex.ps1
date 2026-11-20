# Rebuild dist\SAEDIndex\SAEDIndex.exe.
# PATH is reduced to system folders so unrelated DLLs (for example an old ICU from
# another toolchain) cannot be bundled and break QtCore loading.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$base = (& $python -c "import sys; print(sys.base_prefix)").Trim()
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$base;$(Split-Path $python)"
$env:PYTHONIOENCODING = "utf-8"
& $python -m PyInstaller --noconfirm --clean --log-level WARN SAEDIndex.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
Write-Host "Built: $PSScriptRoot\dist\SAEDIndex\SAEDIndex.exe"
