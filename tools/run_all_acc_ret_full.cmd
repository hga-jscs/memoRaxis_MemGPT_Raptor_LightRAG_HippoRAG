@echo off
setlocal enabledelayedexpansion

REM === User editable ===
set "PROJECT_ROOT=D:\memoRaxis"
set "CONDA_ROOT=D:\anaconda3"
set "CONDA_ENV=memoraxis"
set "OPENAI_BASE_URL=https://jeniya.top/v1"
set "OPENAI_API_KEY=PUT_YOUR_TOKEN_HERE"
REM =====================

REM --- Start/ensure docker pgvector ---
echo [INFO] Ensuring docker container letta_pg...
docker ps -a --format "{{.Names}}" | findstr /I "^letta_pg$" >nul
if errorlevel 1 (
  docker run --name letta_pg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=letta -p 55432:5432 -d pgvector/pgvector:pg16 || (
    echo [ERR] docker run failed
    exit /b 1
  )
) else (
  docker start letta_pg >nul
)

docker exec -it letta_pg psql -U postgres -d letta -c "CREATE EXTENSION IF NOT EXISTS vector;" >nul

REM --- Start letta server in new window ---
echo [INFO] Starting Letta server in a new terminal window...
start "Letta Server" cmd /k ^
  "cd /d %PROJECT_ROOT% && call %CONDA_ROOT%\Scripts\activate.bat %CONDA_ENV% && ^
   set LETTA_DIR=%PROJECT_ROOT%\.letta && ^
   set LETTA_PG_URI=postgresql://postgres:postgres@127.0.0.1:55432/letta?sslmode=disable && ^
   set OPENAI_BASE_URL=%OPENAI_BASE_URL% && ^
   set OPENAI_API_KEY=%OPENAI_API_KEY% && ^
   letta server"

REM --- Wait health ---
echo [INFO] Waiting for Letta health...
for /l %%i in (1,1,60) do (
  powershell -NoProfile -Command "try { (Invoke-RestMethod 'http://127.0.0.1:8283/v1/health' -TimeoutSec 2) | Out-Null; exit 0 } catch { exit 1 }" >nul
  if not errorlevel 1 goto HEALTH_OK
  timeout /t 1 >nul
)
echo [ERR] Letta server did not become healthy in time.
exit /b 1

:HEALTH_OK
echo [INFO] Letta is healthy. Running full pipeline...
call "%PROJECT_ROOT%\tools\run\run_acc_ret_full.cmd"
