# ED-test

`ed-web-platform` is the browser-facing ED/QMC validation app that lives inside the `ED-test` repository.

## What it does

- Upload an extracted case folder tree or one `.zip` bundle from the browser.
- Reuse the current `energy_benchmark.py` logic to compare `energy` and `Green` observables.
- Persist every run locally under `storage/runs/`.
- Export JSON, CSV, and Markdown reports for each validation run.
- Run unit tests and a sample validation job in GitHub Actions.

## Local run

```powershell
cd D:\Lee\QMC\NNN-LxLy\worktrees\2026-06-03_alt_request_clean\data\ed-web-platform
python -m pip install -e .
python -m uvicorn ed_platform.app:app --app-dir src --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## CLI validation

```powershell
python -m ed_platform.cli tests/fixtures/minimal_case --output-dir tmp-output
```

## Test

```powershell
python -m unittest discover -s tests -v
```

## One-click publish

Run this directly in PowerShell after you finish one round of edits:

```powershell
cd D:\Lee\QMC\NNN-LxLy\worktrees\2026-06-03_alt_request_clean\data\ed-web-platform
.\publish.ps1 "your commit message"
```

If PowerShell execution policy blocks `.ps1`, use:

```powershell
.\publish.bat "your commit message"
```

The script will:

- run unit tests
- stage all changes
- create a git commit
- push to the current branch on GitHub

You can skip tests explicitly with:

```powershell
.\publish.ps1 "your commit message" -SkipTests
```

## Run from GitHub

- `ci.yml` runs on `push`, `pull_request`, and manual dispatch.
- `validate-case.yml` lets you trigger a manual GitHub Actions run against any case path already committed to the repo, then downloads JSON/CSV/Markdown validation artifacts.

## GitHub sync note

This repository is ready to be pushed as a standalone GitHub project. The code and update log sync naturally through normal git commits and pushes. Full automatic export of the live Codex conversation into GitHub would require an authenticated GitHub API or CLI bridge, which is not available in the current local environment yet.
