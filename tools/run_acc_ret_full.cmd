@echo off
setlocal enabledelayedexpansion

REM === User editable ===
set "PROJECT_ROOT=D:\memoRaxis"
set "CONDA_ROOT=D:\anaconda3"
set "CONDA_ENV=memoraxis"
set "LETTA_BASE_URL=http://127.0.0.1:8283/v1"
set "OUTPUT_SUFFIX=letta_full"
REM =====================

call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV%" || (
  echo [ERR] Failed to activate conda env: %CONDA_ENV%
  exit /b 1
)

cd /d "%PROJECT_ROOT%" || (
  echo [ERR] Project root not found: %PROJECT_ROOT%
  exit /b 1
)

REM --- Ensure imports work ---
set "PYTHONPATH=%PROJECT_ROOT%"

REM --- Ensure we do NOT bind to an old agent by env ---
set "LETTA_AGENT_ID="
set "LETTA_BASE_URL=%LETTA_BASE_URL%"

echo [INFO] (1/5) Health check Letta...
powershell -NoProfile -Command "try { (Invoke-RestMethod '%LETTA_BASE_URL%/health' -TimeoutSec 5) | Out-Null; 'ok' } catch { 'fail' }" | findstr /I ok >nul || (
  echo [ERR] Letta server not ready at %LETTA_BASE_URL%. Start it first (start_letta_server.cmd).
  exit /b 1
)

echo [INFO] (2/5) Ingest Accurate_Retrieval instance 0 (1204 chunks)...
python scripts\simpleMemory_MAB\ingest\ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 850 || (
  echo [ERR] ingest failed
  exit /b 1
)

echo [INFO] (3/5) Ensure preview_samples JSON exists...
python scripts\simpleMemory_MAB\data\convert_parquet_to_json.py || (
  echo [ERR] convert_parquet_to_json failed
  exit /b 1
)

echo [INFO] (4/5) Infer (FULL) R1/R2/R3...
python scripts\simpleMemory_MAB\infer\infer_accurate_retrieval.py --instance_idx 0 --adaptor all --limit -1 --output_suffix %OUTPUT_SUFFIX% || (
  echo [ERR] infer failed
  exit /b 1
)

REM --- Resolve results file path robustly (avoid hardcoded name mismatch) ---
for /f "usebackq delims=" %%I in (`python -c "import glob; fs=sorted(glob.glob('out/*acc*ret*results*0*%OUTPUT_SUFFIX%*.json')); print(fs[-1] if fs else '')"`) do set "RESULTS_JSON=%%I"

if "%RESULTS_JSON%"=="" (
  echo [ERR] Cannot find results json under out/ with suffix=%OUTPUT_SUFFIX%
  dir out
  exit /b 1
)

echo [INFO] results json: %RESULTS_JSON%

echo [INFO] (5/5) Evaluate + Analyze...
python scripts\simpleMemory_MAB\evaluate\evaluate_mechanical.py --results "%RESULTS_JSON%" --instance MemoryAgentBench\preview_samples\Accurate_Retrieval\instance_0.json > acc_ret_summary_raw.txt || (
  echo [ERR] evaluate_mechanical failed
  exit /b 1
)

python scripts\simpleMemory_MAB\analyze\analyze_acc_ret.py > out\acc_ret_report.txt || (
  echo [ERR] analyze_acc_ret failed
  exit /b 1
)

echo.
echo [DONE] Artifacts:
echo   - %RESULTS_JSON%
echo   - acc_ret_summary_raw.txt
echo   - out\acc_ret_report.txt
echo.
type out\acc_ret_report.txt
