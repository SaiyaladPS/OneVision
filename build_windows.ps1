$ErrorActionPreference = "Stop"

# Run from the repository root after installing: pip install -e ".[build]"
python -m PyInstaller --noconfirm --clean car_scan.spec
Write-Host "Build complete: dist\CarScan\CarScan.exe"

# Optional: install Inno Setup, then build a conventional Windows installer.
$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($iscc) {
    & $iscc.Source installer\CarScan.iss
    Write-Host "Installer complete: installer\output\CarScan-Setup.exe"
} else {
    Write-Host "ISCC.exe not found; the portable folder build is ready."
}
