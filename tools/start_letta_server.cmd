@echo off
setlocal enabledelayedexpansion

REM === User editable ===
set "PROJECT_ROOT=D:\memoRaxis"
set "CONDA_ROOT=D:\anaconda3"
set "CONDA_ENV=memoraxis"

set "LETTA_DIR=%PROJECT_ROOT%\.letta"
set "LETTA_PG_URI=postgresql://postgres:postgres@127.0.0.1:55432/letta?sslmode=disable"

REM OpenAI-compat gateway (jeniya)
set "OPENAI_BASE_URL=https://jeniya.top/v1"
set "OPENAI_API_KEY=PUT_YOUR_TOKEN_HERE"
REM =====================

call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV%" || (
  echo [ERR] Failed to activate conda env: %CONDA_ENV%
  exit /b 1
)

cd /d "%PROJECT_ROOT%" || (
  echo [ERR] Project root not found: %PROJECT_ROOT%
  exit /b 1
)

set "LETTA_DIR=%LETTA_DIR%"
set "LETTA_PG_URI=%LETTA_PG_URI%"
set "OPENAI_BASE_URL=%OPENAI_BASE_URL%"
set "OPENAI_API_KEY=%OPENAI_API_KEY%"

echo [INFO] Starting Letta server...
echo [INFO] LETTA_PG_URI=%LETTA_PG_URI%
echo [INFO] OPENAI_BASE_URL=%OPENAI_BASE_URL%
echo [INFO] LETTA_DIR=%LETTA_DIR%

letta server
