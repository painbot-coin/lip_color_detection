@echo off
REM Run the lipcolor CLI with the correct Python (py311 has mediapipe; base 3.13 does not).
REM Usage:  run face.jpg [more.jpg ...] [--json] [--debug overlays\]
setlocal
set "PY=C:\Users\Admin\anaconda3\envs\py311\python.exe"
if not exist "%PY%" (
  echo [run.bat] py311 interpreter not found at %PY%
  echo           Edit run.bat and set PY to your py311 python.exe, or: conda activate py311
  exit /b 1
)
"%PY%" -m lipcolor.cli %*
endlocal
