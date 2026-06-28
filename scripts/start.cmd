@echo off
rem ============================================================================
rem  Audicop launcher for Windows -- double-click this file.
rem
rem  Why a .cmd wrapper: corporate machines often block unsigned PowerShell
rem  *files* (AllSigned/RemoteSigned, sometimes via Group Policy), and files
rem  under OneDrive get a "downloaded from internet" mark. Those block
rem  `start.ps1` even with `-ExecutionPolicy Bypass`. This wrapper runs the same
rem  launcher INLINE (`-Command`), which the execution policy does NOT restrict
rem  (it only restricts running .ps1 files), so it works everywhere.
rem ============================================================================
setlocal
set "AUDICOP_SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:AUDICOP_SCRIPT_DIR='%~dp0'; Invoke-Expression (Get-Content -Raw -LiteralPath '%~dp0start.ps1')"
if errorlevel 1 (
  echo.
  echo Audicop termino con un error. Revisa el mensaje de arriba.
  pause
)
