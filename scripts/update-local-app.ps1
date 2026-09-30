<#
.SYNOPSIS
  Brings this checkout up to a branch, rebuilds the native ARJUN app (the
  executable only, no installer), and puts a desktop shortcut to it.

.DESCRIPTION
  Run it yourself, on the Windows machine, from a PowerShell prompt in the
  repository root:

      powershell -ExecutionPolicy Bypass -File scripts\update-local-app.ps1

  What it does, in order, and nothing else:
    1. git fetch + fast-forward to -Branch (skip with -NoFetch). This is the
       only step that reaches the network, and only to your own git remote.
    2. Builds the agent runtime bundle (npm run runtime:build).
    3. Builds the app with the llama.cpp backend this machine supports
       (node scripts/select-backend.mjs build --no-bundle): CUDA, Vulkan or
       CPU, chosen by probing, as `npm run dev:auto` does.
    4. Creates "<Desktop>\<ShortcutName>.lnk" pointing at the built exe.

  It installs no package, downloads no model and runs no installer. If your
  node_modules are missing, install them first (npm ci, and
  npm run runtime:install), deliberately.

.PARAMETER Branch
  The branch to build. Default: the branch this work was pushed to.

.PARAMETER ShortcutName
  The shortcut's name on the desktop. Default: "arjun new".

.PARAMETER NoFetch
  Build the checkout as it is, without fetching.
#>
param(
    [string]$Branch = "claude/hopeful-brahmagupta-1eol9d",
    [string]$ShortcutName = "arjun new",
    [switch]$NoFetch
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($what) { Write-Host "`n== $what" -ForegroundColor Cyan }

if (-not $NoFetch) {
    Step "Updating the checkout to $Branch"
    $dirty = git status --porcelain
    if ($dirty) {
        throw "The checkout has uncommitted changes. Commit or stash them first; nothing was changed."
    }
    git fetch origin $Branch
    git checkout $Branch
    git merge --ff-only "origin/$Branch"
}

foreach ($needed in @("node_modules", "agent-runtime\node_modules")) {
    if (-not (Test-Path (Join-Path $root $needed))) {
        throw "$needed is missing. Install the dependencies first (npm ci; npm run runtime:install), then run this again."
    }
}

Step "Building the agent runtime bundle"
npm run runtime:build
if ($LASTEXITCODE -ne 0) { throw "npm run runtime:build failed (exit $LASTEXITCODE)." }

Step "Building the native app (executable only, no installer)"
node scripts/select-backend.mjs build --no-bundle
if ($LASTEXITCODE -ne 0) { throw "The app build failed (exit $LASTEXITCODE)." }

# Tauri names the release executable after the product; older builds after
# the crate. Whichever this build wrote most recently is the app.
$release = Join-Path $root "src-tauri\target\release"
$exe = Get-ChildItem -Path $release -Filter *.exe -File |
    Where-Object { $_.BaseName -in @("ARJUN", "arjun", "sarathi") } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $exe) {
    throw "The build finished but no ARJUN executable was found in $release."
}

Step "Creating the desktop shortcut"
$desktop = [Environment]::GetFolderPath("Desktop")
$link = Join-Path $desktop "$ShortcutName.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $exe.FullName
$shortcut.WorkingDirectory = $exe.DirectoryName
$shortcut.IconLocation = "$($exe.FullName),0"
$shortcut.Description = "ARJUN, built from $Branch"
$shortcut.Save()

Write-Host "`nBuilt:    $($exe.FullName)" -ForegroundColor Green
Write-Host "Shortcut: $link" -ForegroundColor Green
