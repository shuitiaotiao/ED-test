@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0publish.ps1" %*
endlocal
