@echo off
rem ---------------------------------------------------------------------------
rem Start smDA-HMM in your browser.
rem
rem Double-click this file, or run it from a command prompt.
rem
rem It uses the .venv beside this script and refuses to fall back to whatever
rem Python happens to be on PATH.  smDA-Python's start_smda.bat did fall back,
rem which meant it launched against an interpreter without the dependencies and
rem failed with "No module named 'smda'" -- a confusing way to be told the
rem environment was never set up.  If .venv is missing this script says so and
rem gives the two commands that create it.
rem
rem Port 8502 keeps this separate from smDA-Python on 8501, so both can run.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo.
    echo   No virtual environment found at:
    echo     %VENV_PY%
    echo.
    echo   Create it once with:
    echo     py -3.11 -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    echo   Then build the Rust extension - see README.md, "Running from source".
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0smda_scan\python\smda_scan\smda_scan.cp311-win_amd64.pyd" (
    echo.
    echo   The Rust extension is not built. Build it once with:
    echo     cd smda_scan
    echo     cargo build --release
    echo     copy target\release\smda_scan.dll python\smda_scan\smda_scan.cp311-win_amd64.pyd
    echo.
    echo   See README.md, "Running from source".
    echo.
    pause
    exit /b 1
)

rem PYTHONPATH is not needed: app.py puts the package and the extension on
rem sys.path itself.  Streamlit only adds the script's own directory, which is
rem why relying on that alone would not be enough.
echo Starting smDA-HMM on http://localhost:8502
"%VENV_PY%" -m streamlit run app.py --server.port 8502

endlocal
