param(
    [Parameter(Position = 0)]
    [string]$Message,

    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not $Message) {
    $Message = Read-Host "Commit message"
}

if (-not $Message) {
    throw "A commit message is required."
}

$branch = (& git branch --show-current).Trim()
if (-not $branch) {
    throw "Could not determine the current git branch."
}

$statusLines = @(& git status --short)
if (-not $statusLines -or $statusLines.Count -eq 0) {
    Write-Host "No local changes to publish." -ForegroundColor Yellow
    exit 0
}

Write-Host "Repository: $repoRoot" -ForegroundColor Cyan
Write-Host "Branch: $branch" -ForegroundColor Cyan
Write-Host "Pending changes:" -ForegroundColor Cyan
$statusLines | ForEach-Object { Write-Host "  $_" }

if (-not $SkipTests) {
    Write-Host ""
    Write-Host "Running unit tests..." -ForegroundColor Green
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed. Publish aborted."
    }
}

Write-Host ""
Write-Host "Staging changes..." -ForegroundColor Green
Invoke-Git -Args @("add", "-A")

$stagedStatus = @(& git diff --cached --name-only)
if (-not $stagedStatus -or $stagedStatus.Count -eq 0) {
    Write-Host "Nothing staged after git add -A." -ForegroundColor Yellow
    exit 0
}

Write-Host "Creating commit..." -ForegroundColor Green
Invoke-Git -Args @("commit", "-m", $Message)

$hasUpstream = $true
& git rev-parse --abbrev-ref --symbolic-full-name "@{u}" *> $null
if ($LASTEXITCODE -ne 0) {
    $hasUpstream = $false
}

Write-Host "Pushing to GitHub..." -ForegroundColor Green
if ($hasUpstream) {
    Invoke-Git -Args @("push")
} else {
    Invoke-Git -Args @("push", "-u", "origin", $branch)
}

Write-Host ""
Write-Host "Publish complete." -ForegroundColor Green
& git log --oneline --decorate -1
