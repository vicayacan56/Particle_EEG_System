<#
Simple helper to bootstrap vcpkg and install required packages (Windows PowerShell)
Usage: .\scripts\install_vcpkg.ps1
#>
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repo = Resolve-Path "$root\.."

Write-Host "Repo root: $repo"

$vcpkgDir = Join-Path $repo 'vcpkg'
if (-Not (Test-Path $vcpkgDir)) {
    Write-Host "Cloning vcpkg into $vcpkgDir"
    git clone https://github.com/microsoft/vcpkg.git $vcpkgDir
}

Push-Location $vcpkgDir
if (-Not (Test-Path "$vcpkgDir\vcpkg.exe")) {
    Write-Host "Bootstrapping vcpkg..."
    & .\bootstrap-vcpkg.bat
}

Write-Host "Integrating vcpkg with Visual Studio (may require admin)..."
.\vcpkg integrate install | Out-Null

Write-Host "Installing packages: glew, freeglut"
.\vcpkg install glew freeglut --triplet x64-windows

Pop-Location
Write-Host "vcpkg ready. If you run into issues, run this script as Administrator."
