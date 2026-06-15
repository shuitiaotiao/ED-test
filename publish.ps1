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

function Get-HasUpstream {
    & git rev-parse --abbrev-ref --symbolic-full-name "@{u}" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-AheadCount {
    param(
        [bool]$HasUpstream
    )

    if ($HasUpstream) {
        $countText = (& git rev-list --count "@{u}..HEAD").Trim()
        if (-not $countText) {
            return 0
        }
        return [int]$countText
    }

    & git rev-parse --verify HEAD *> $null
    if ($LASTEXITCODE -ne 0) {
        return 0
    }

    return 1
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$branch = (& git branch --show-current).Trim()
if (-not $branch) {
    throw "Could not determine the current git branch."
}

$hasUpstream = Get-HasUpstream
$aheadCount = Get-AheadCount -HasUpstream $hasUpstream
$statusLines = @(& git status --short)

Write-Host "Repository: $repoRoot" -ForegroundColor Cyan
Write-Host "Branch: $branch" -ForegroundColor Cyan
if ($hasUpstream) {
    Write-Host "Unpushed commits: $aheadCount" -ForegroundColor Cyan
} else {
    Write-Host "No upstream branch configured yet." -ForegroundColor Yellow
}

if (-not $statusLines -or $statusLines.Count -eq 0) {
    if ($aheadCount -gt 0) {
        Write-Host "No uncommitted changes. Pushing existing local commits..." -ForegroundColor Green
        if ($hasUpstream) {
            Invoke-Git -Args @("push")
        } else {
            Invoke-Git -Args @("push", "-u", "origin", $branch)
        }
        Write-Host ""
        Write-Host "Publish complete." -ForegroundColor Green
        & git log --oneline --decorate -1
        exit 0
    }

    Write-Host "No local changes or unpublished commits." -ForegroundColor Yellow
    exit 0
}

Write-Host "Pending changes:" -ForegroundColor Cyan
$statusLines | ForEach-Object { Write-Host "  $_" }

if (-not $Message) {
    $Message = Read-Host "Commit message"
}

if (-not $Message) {
    throw "A commit message is required."
}

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

Write-Host "Pushing to GitHub..." -ForegroundColor Green
if ($hasUpstream) {
    Invoke-Git -Args @("push")
} else {
    Invoke-Git -Args @("push", "-u", "origin", $branch)
}

Write-Host ""
Write-Host "Publish complete." -ForegroundColor Green
& git log --oneline --decorate -1
